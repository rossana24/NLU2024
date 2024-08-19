from collections import Counter
import json
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.utils.data as data

PAD_TOKEN = 0


def load_data(path):
    """
    It is a simple method that you laod everything in format of json and return it
     * input: path/to/data
     * output: json
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids


class Lang(object):
    """
    Handles vocabulary and label-to-ID mappings for intents and slots.

    Attributes:
        pad_token (int): ID for the padding token.
        slot2id (dict): Mapping from slot labels to IDs.
        intent2id (dict): Mapping from intent labels to IDs.
        id2slot (dict): Reverse mapping from IDs to slot labels.
        id2intent (dict): Reverse mapping from IDs to intent labels.
        vocab (set): Set of vocabulary words belonging to the training set.
    """
    def __init__(self, word, intents, slots, pad_token):
        self.pad_token = pad_token
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.vocab = set(word)

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab

    def save(self, path):
        torch.save({
            'vocab': self.vocab,
            'slot2id': self.slot2id,
            'intent2id': self.intent2id,
            'pad_token': self.pad_token
        }, path)

    @classmethod
    def load(cls, checkpoint):
        instance = cls(
            word=checkpoint['vocab'],
            intents=checkpoint['intent2id'].keys(),
            slots=checkpoint['slot2id'].keys(),
            pad_token=checkpoint['pad_token']
        )

        instance.vocab = checkpoint['vocab']
        instance.slot2id = checkpoint['slot2id']
        instance.intent2id = checkpoint['intent2id']
        instance.pad_token = checkpoint['pad_token']
        instance.id2slot = {v: k for k, v in instance.slot2id.items()}
        instance.id2intent = {v: k for k, v in instance.intent2id.items()}
        return instance

    def print_all_slots(self):
        print("Slots and their IDs:")
        for slot, slot_id in self.slot2id.items():
            print(f"{slot}: {slot_id}")

    def print_all_intents(self):
        print("Intents and their IDs:")
        for intent, intent_id in self.intent2id.items():
            print(f"{intent}: {intent_id}")


class IntentsAndSlots(data.Dataset):
    """
    A custom dataset class for handling intents and slots.

    Args:
        dataset (list): List of raw data samples.
        lang (Lang): Language processing object.
        tokenizer (BertTokenizer): BERT tokenizer instance.
    """
    def __init__(self, dataset, lang, tokenizer):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.lang = lang

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
        self.final_features = self.convert_text_to_features()

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        feature = self.final_features[idx]
        utterance_tensor = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_tensor = torch.tensor(feature.token_type_ids, dtype=torch.long)
        intent_type_tensor = torch.tensor(feature.intent_label_id, dtype=torch.long)
        slot_type_tensor = torch.tensor(feature.slot_labels_ids, dtype=torch.long)

        sample = {'utterance': utterance_tensor,
                  'slots': slot_type_tensor,
                  'intent': intent_type_tensor,
                  'attention': attention_mask_tensor,
                  'token_type': token_type_tensor
                  }
        return sample

    # Auxiliary methods
    def convert_text_to_features(self):
        """
        Convert raw text data into features suitable for BERT input.

        Output:
         * list: List of InputFeatures objects.
        """
        # Setting based on the current model type
        cls_token = self.tokenizer.cls_token  # [CLS]
        sep_token = self.tokenizer.sep_token  # [SEP]
        unk_token = self.tokenizer.unk_token  # unknown token

        sequence_a_segment_id = 0
        cls_token_segment_id = 0
        features = []
        for index, (utterance, slot_id, intent_id) in enumerate(zip(self.utterances, self.slot_ids, self.intent_ids)):
            tokens = []
            slot_labels_ids = []
            for word, slot_label in zip(utterance.split(), slot_id):
                if word in self.lang.vocab:
                    word_tokens = self.tokenizer.tokenize(text=word)
                    # Ensure unknown tokens in dev and test sets are replaced with UNK token
                    if not word_tokens:
                        word_tokens = [unk_token]
                    tokens.extend(word_tokens)
                else:
                    word_tokens = [unk_token]
                    tokens.extend(word_tokens)

                slot_labels_ids.extend([slot_label] + [self.lang.slot2id['pad']] * (len(word_tokens) - 1))
                # Padding the slot labels aligned with sub-words resulting from self.tokenizer.tokenize(text=word)
                # Example:
                # Original sentence: I like your kindness
                # tokenization by bert: I like your kind ##ness
                # padding slot labels: O O O B-loc <pad>

            # Add [SEP] token
            tokens += [sep_token]
            slot_labels_ids += [self.lang.slot2id['pad']]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            slot_labels_ids = [self.lang.slot2id['pad']] + slot_labels_ids

            token_type_ids = [cls_token_segment_id] + token_type_ids
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            assert len(tokens) == len(
                slot_labels_ids), f"Length mismatch: tokens ({len(tokens)}) != slot_labels_ids ({len(slot_labels_ids)})"
            assert len(tokens) == len(
                attention_mask), f"Length mismatch: input_ids ({len(input_ids)}) != slot_labels_ids ({len(slot_labels_ids)}) "

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              intent_label_id=intent_id,
                              slot_labels_ids=slot_labels_ids
                              ))

        return features

    @staticmethod
    def mapping_lab(data, mapper):
        return [mapper[x] for x in data]

    @staticmethod
    def mapping_seq(data, mapper):  # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                tmp_seq.append(mapper[x])
            res.append(tmp_seq)
        return res


class CollateFn:

    def __init__(self, word_pad_token, slot_pad_token, attention_pad_token, intent_label_id_padding, device):
        self.word_pad_token = word_pad_token
        self.slot_pad_token = slot_pad_token
        self.attention_pad_token = attention_pad_token
        self.intent_label_id_padding = intent_label_id_padding
        self.device = device

    def __call__(self, data_dl):
        def merge(sequences, pad_token):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq
            padded_seqs = padded_seqs.detach()
            return padded_seqs, lengths

        data_dl.sort(key=lambda x: len(x['utterance']), reverse=True)
        new_item = {}

        # Creates a new dictionary new_item where each key corresponds to a list of values from all samples.
        for key in data_dl[0].keys():
            new_item[key] = [d[key] for d in data_dl]
        src_utt, _ = merge(new_item['utterance'], self.word_pad_token)
        y_slots, y_lengths = merge(new_item["slots"], self.slot_pad_token)
        src_attention, _ = merge(new_item['attention'], self.attention_pad_token)
        y_token_type, _ = merge(new_item["token_type"], self.intent_label_id_padding)
        intent = torch.LongTensor(new_item["intent"])

        # Move tensors to the specified device
        src_utt = src_utt
        y_slots = y_slots
        intent = intent
        src_attention = src_attention
        y_token_type = y_token_type
        y_lengths = torch.LongTensor(y_lengths)

        new_item["utterance"] = src_utt
        new_item["intents"] = intent
        new_item["slots"] = y_slots
        new_item["attention"] = src_attention
        new_item["token_type"] = y_token_type
        new_item["slots_len"] = y_lengths
        return new_item


def get_data_loaders(args, load_lang=None):
    """
     Load dataset, split into training and validation sets, and prepare data loaders.

     Args:
      * path (str): Path to the dataset file.
      * tokenizer (BertTokenizer): BERT tokenizer instance.
      * lang (Lang): Language processing object.
      * batch_size (int): Size of each batch.
      * test_size (float): Proportion of the dataset to be used for validation.
     """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This code is running on :{device}")

    tmp_train_raw = load_data(args.Dataset.train_path)
    test_raw = load_data(args.Dataset.test_path)
    print("Size of dataset before splitting validation set")
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    # Define the proportion of the training set to be used for validation
    portion = args.Dataset.portion

    labels = []
    inputs = []
    mini_train = []

    # Separate intents that occur only once to ensure they are included in the training set
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            # Include intents with more than one occurrence in the training set
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            # Include intents that occur only once in the mini_train set
            mini_train.append(tmp_train_raw[id_y])

    # Perform stratified splitting of the training set into training and validation sets
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                      random_state=args.Training.seed,
                                                      shuffle=True,
                                                      stratify=labels)

    # Add samples with rare intents back to the training set
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    # Print dataset sizes after splitting
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))

    # Initialize the BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create a corpus by combining training, validation, and test data
    corpus = train_raw + dev_raw + test_raw  # full dataset
    max_length = max([len(item['utterance'].split()) for item in corpus])
    print(f"The longest length of a sentence is {max_length}")

    # Extract vocabulary, slots, and intents
    words = sum([x['utterance'].split() for x in train_raw], [])  # No set() since we want to compute
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    if load_lang is None:
        dataset_processor = Lang(words, intents, slots, args.Model.pad_id)
    else:
        dataset_processor = load_lang

    train_dataset = IntentsAndSlots(train_raw, dataset_processor, bert_tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, dataset_processor, bert_tokenizer)
    test_dataset = IntentsAndSlots(test_raw, dataset_processor, bert_tokenizer)

    sampler_train = RandomSampler(train_dataset)
    sampler_dev = SequentialSampler(dev_dataset)
    sampler_test = SequentialSampler(test_dataset)

    # Define padding tokens for different data types
    pad_token_slot = dataset_processor.slot2id['pad']
    pad_token_utterance = bert_tokenizer.pad_token_id
    pad_attention = 0
    pad_id = 0

    collate_fn = CollateFn(pad_token_utterance, pad_token_slot, pad_attention, pad_id, device)

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, sampler=sampler_train, batch_size=args.Dataset.batch_size_train,
                              collate_fn=collate_fn, num_workers=1)
    dev_loader = DataLoader(dev_dataset, sampler=sampler_dev, batch_size=args.Dataset.batch_size_valid,
                            collate_fn=collate_fn, num_workers=1)
    test_loader = DataLoader(test_dataset, sampler=sampler_test, batch_size=args.Dataset.batch_size_test,
                             collate_fn=collate_fn, num_workers=1)

    return train_loader, dev_loader, test_loader, dataset_processor, bert_tokenizer
