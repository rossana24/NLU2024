# Add functions or classes used for data loading and preprocessing

import json
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


PAD_TOKEN = 0


class Lang:
    """
    Summary:
    This class provides a mapping from words, slots, and intents to integer IDs and vice versa.
    It builds vocabularies for words, slots, and intents, and handles their conversion.
    """
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

    def save(self, path):
        torch.save({
            'word2id': self.word2id,
            'slot2id': self.slot2id,
            'intent2id': self.intent2id,
        }, path)

    @classmethod
    def load(cls, checkpoint):
        instance = cls(
            words=checkpoint['word2id'].keys(),
            intents=checkpoint['intent2id'].keys(),
            slots=checkpoint['slot2id'].keys(),
        )
        instance.word2id = checkpoint['word2id']
        instance.slot2id = checkpoint['slot2id']
        instance.intent2id = checkpoint['intent2id']
        instance.id2word = {v: k for k, v in instance.word2id.items()}
        instance.id2slot = {v: k for k, v in instance.slot2id.items()}
        instance.id2intent = {v: k for k, v in instance.intent2id.items()}
        return instance


def get_data_loaders(args, loaded_lang=None):
    """
    Summary:
    Loads and processes the dataset, splits it into training, validation, and test sets,
    and creates data loaders for these sets.

    Input:
     * args (OmegaConf object): Configuration parameters including dataset paths and split ratio.

    Output:
     * train_loader (DataLoader): DataLoader for the training dataset.
     * dev_loader (DataLoader): DataLoader for the validation dataset.
     * test_loader (DataLoader): DataLoader for the test dataset.
     * lang (Lang): Language object containing mappings for words, slots, and intents.
    """
    tmp_train_raw = load_data(args.Dataset.train_path)
    test_raw = load_data(args.Dataset.test_path)
    print("Size of dataset before splitting validation set")
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    # Split dataset into training and validation sets
    portion = args.Dataset.portion
    intents = [x['intent'] for x in tmp_train_raw]

    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                      random_state=args.Training.seed,
                                                      shuffle=True,
                                                      stratify=labels)
    '''
    The stratify parameter in the train_test_split function from the sklearn.model_selection module ensures that the
    train and test sets have the same proportion of samples for each class label as the original dataset. 
    Stratified Sampling
    Stratified sampling involves dividing the dataset into distinct subgroups (strata) and then randomly sampling from 
    each subgroup in a way that the train and test sets reflect the original distribution of the labels.
    Purpose
    The main purpose of using stratify is to maintain the class distribution of the dataset in both the train and test 
    sets. This is particularly important in the following scenarios:
    Imbalanced Datasets: When you have an imbalanced dataset where certain classes are underrepresented, 
    stratifying ensures that these classes are adequately represented in both the train and test sets.
    Consistency: It provides consistency in the class distribution, which can lead to more reliable and generalizable 
    model performance.
    '''
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw  # We do not wat unk labels, however this depends on the research purpose

    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    # creating the language model
    if loaded_lang is None:
        lang = Lang(words, intents, slots, cutoff=0)
    else:
        lang = loaded_lang
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    sampler_train = RandomSampler(train_dataset)
    sampler_dev = SequentialSampler(dev_dataset)
    sampler_test = SequentialSampler(test_dataset)
    train_loader = DataLoader(train_dataset, sampler=sampler_train, batch_size=args.Dataset.batch_size_train,
                              collate_fn=collate_fn, num_workers=2)
    dev_loader = DataLoader(dev_dataset, sampler=sampler_dev, batch_size=args.Dataset.batch_size_valid,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, sampler= sampler_test, batch_size=args.Dataset.batch_size_test,
                             collate_fn=collate_fn, num_workers=2)

    return train_loader, dev_loader, test_loader, lang


class IntentsAndSlots(data.Dataset):
    """
    Summary:
    Custom dataset class to handle utterances, slots, and intents, and map them to numerical IDs.
    It implements the mandatory methods for PyTorch datasets.

    Input:
     * dataset (list of dict): List of data samples containing 'utterance', 'slots', and 'intent'.
     * lang (Lang): Language object with mappings for words, slots, and intents.
     * unk (str): Token used for unknown words.
    """

    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def load_data(path):
    """
    Summary:
    Loads dataset from a JSON file.

    Input:
     * path (str): Path to the JSON file.

    Output:
     * dataset (list of dict): List of data samples loaded from the file.
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def collate_fn(data):
    """
    Summary:
    Custom collate function for batching sequences with variable lengths.

    Input:
     * data (list of dict): List of data samples where each sample is a dictionary containing 'utterance',
       'slots', and 'intent'.

    Output:
     * new_item (dict): Dictionary containing batched and padded sequences along with their original lengths.
    """

    def merge(sequences):
        """
        Pads sequences to the same length within a batch.

        Input:
         * sequences (list of list of int): List of sequences to be padded.

        Output:
         * padded_seqs (Tensor): Tensor of padded sequences.
         * lengths (list of int): List of original lengths of sequences.
        """

        lengths = [len(seq) for seq in sequences]
        # Finds the maximum length in the batch.
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        # Creates a tensor filled with PAD_TOKEN (0 in this case) of shape (batch_size, max_length).
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort data by the seq length of 'utterance' in descending order
    data.sort(key=lambda x: len(x['utterance']), reverse=True)

    new_item = {}
    # Creates a new dictionary new_item where each key corresponds to a list of values from all samples
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt
    y_slots = y_slots
    intent = intent
    y_lengths = torch.LongTensor(y_lengths)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

