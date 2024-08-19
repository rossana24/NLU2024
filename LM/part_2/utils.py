# Add functions or classes used for data loading and preprocessing


import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader


def get_data_loaders(args, load_lang=None):
    """
    Summary
    This function that returns the dataloaders for the test, train and validation set
    Input:
     * args (dict): Dictionary containing all the paramters of the use
    Output:
     * train_loader (Dataloader): This is the dataloader for the training set
     * dev_loader (Dataloader) : This is the dataloader for the training set
     * test_loader (Dataloader): This is the dataloader for the training set
     * lang (Language): This is a class used for processing managing vocab in NLP>
    """
    train_raw = read_file(args.Dataset.train_dataset_path)
    dev_raw = read_file(args.Dataset.valid_dataset_path)
    test_raw = read_file(args.Dataset.test_dataset_path)

    '''
    After padding with <pad>:
        Padded sentence 1: "This is a short sentence<pad><pad><pad><pad><pad>" (length: 10)
        Padded sentence 2: "This is a longer sentence<pad><pad><pad>" (length: 10)
    '''
    if load_lang is None:
        lang = Lang(train_raw, ["<pad>", "<eos>"])
    else:
        lang = load_lang

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=args.Dataset.batch_size_train,
                              collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True,
                              drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.Dataset.batch_size_valid,
                            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"])
                            , drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.Dataset.batch_size_test,
                             collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
                             drop_last=False)

    return train_loader, dev_loader, test_loader, lang


def read_file(path, eos_token="<eos>"):
    """
    Summary
    This function reads a file line by line, cleans each line by removing whitespaces,
     and adds a user-defined end-sentence token to each line. Finally, it returns a list of these processed strings.
    Input:
     * path (string): path indicate the file path
     * eos_token (string): indicate token to put in the end of sentence
    Output:
     * output (List): list of sentence ending with eos token
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class Lang:
    def __init__(self, corpus, special_tokens=None):
        """
        Summary
        This is a class used for processing managing vocab in NLP>
        Input:
         * corpus (list):  list of strings, where each string represents a sentence or sequence of words
         * special_tokens (list): some special tokens
        """
        if special_tokens is None:
            special_tokens = []
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=None):
        """
        Summary
        This is a function that give you count of every word in training set
        Input:
         * corpus (list):  list of strings, where each string represents a sentence or sequence of words
         * special_tokens (list): some special tokens
        Output:
         *  output (dict): It counts how many of each word exists.
        """
        if special_tokens is None:
            special_tokens = []
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

    @classmethod
    def load(cls, checkpoint):
        instance = cls(corpus=checkpoint["word2id"].keys(), special_tokens=["<pad>", "<eos>"])
        instance.word2id = checkpoint["word2id"]
        instance.id2word = {v: k for k, v in instance.word2id.items()}
        return instance


class PennTreeBank(data.Dataset):

    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        # Creation of source and target sequences
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for word in seq:
                if word in lang.word2id:
                    tmp_seq.append(lang.word2id[word])
                else:
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res


def collate_fn(data, pad_token):
    """
    Summary:
    The provided collate_fn function is designed to work with a PyTorch dataloader for NLP tasks.
    It addresses the challenge of dealing with variable-length sequences in text data.
    Purpose:
    This function takes a batch of data (data) and a padding token (pad_token) as input.
    It processes the data to ensure all sequences (sentences) within the batch have the same length.
    This is crucial for many NLP models, which require fixed-size inputs.

    Why padding is needed:
        * Real-world text data consists of sentences with varying lengths.
        * However, many NLP models operate on sequences of a specific length.
        * Without padding, a dataloader wouldn't be able to create batches with
        sequences of different lengths.
    Input:
     * data (list) : list of dictionaries containing source and target  for sentence already turned to id
     *  pad_token (list) : list of padding that we have
    Output:
     * sample dict(Long Tensor, Long Tensor): source and target information
    """

    def merge(sequences):
        """
        Summary :
        merge from batch * sent_len to batch * max_len
        Input :
         * sequence(list): this is a list of tensors containing difference sentence in the batch
                           (sorted form the longest to the shorted)
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)

        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)

    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    source, source_lengths = merge(new_item["source"])
    target, target_lengths = merge(new_item["target"])

    new_item["source"] = source
    new_item["target"] = target
    new_item["number_tokens"] = sum(target_lengths)
    new_item["source_lengths"] = torch.tensor(source_lengths, dtype=torch.int64)
    new_item["target_lengths"] = torch.tensor(target_lengths, dtype=torch.int64)

    # Total number of tokens in the batch (including padding).
    return new_item


def tbptt_split_batch(batch_dict, mu, std, p, pad_value):
    """
    Summary:
    This function implements Truncated Backpropagation Through Time (TBPTT) for a batch of sequences.
    It splits the input sequences into smaller chunks to control the length of backpropagation.

    Input:
     * batch_dict (dict): A dictionary containing the batch data, including 'source', 'target',
       'source_lengths', 'target_lengths', and 'number_tokens'.
     * mu (float): Mean of the Gaussian distribution used to determine the chunk size.
     * std (float): Standard deviation of the Gaussian distribution used to determine the chunk size.
     * p (float): Probability of using the mean value as the chunk size.
     * pad_value (int): The padding value used to fill sequences to the same length.

    Output:
     * split_batches (list): A list of dictionaries, where each dictionary represents a chunk of
       sequences with corresponding 'source', 'target', 'source_lengths', 'target_lengths',
       and 'number_tokens'.
    """

    def get_split_step(mu, std, p):
        """
        Summary:
        Determines the length of each chunk for splitting the sequences.
        The step size is sampled from a Gaussian distribution with the specified mean (mu)
        and standard deviation (std).

        Input:
         * mu (float): Mean of the Gaussian distribution.
         * std (float): Standard deviation of the Gaussian distribution.
         * p (float): Probability of using the mean value as the step size.

        Output:
         * split_step (int): The determined chunk size for splitting the sequences.
        """
        mu = mu if np.random.random() < p else mu / 2
        split_step = max(10, int(np.random.normal(mu, std)))
        return split_step

    # Get the split step (chunk size) based on the provided parameters
    split_step = get_split_step(mu, std, p)
    inputs = batch_dict['source']
    targets = batch_dict['target']

    # Split inputs and targets into chunks of split_step
    inputs = list(torch.split(inputs, split_step, dim=1))
    targets = list(torch.split(targets, split_step, dim=1))

    # Function to calculate the number of non-PAD tokens in a sequence
    get_length = lambda x: torch.sum(x.ne(pad_value)).item()

    valid_inputs = []
    valid_targets = []
    valid_number_tokens = []

    for split_idx in range(len(inputs)):
        # Filter out empty sequences from inputs and targets
        filtered_inputs = [i for i in inputs[split_idx] if get_length(i) > 0]
        filtered_targets = [t for t in targets[split_idx] if get_length(t) > 0]

        # If the chunk has valid sequences, store it along with the corresponding token count
        if filtered_inputs:
            valid_inputs.append(torch.stack(filtered_inputs))
            valid_targets.append(torch.stack(filtered_targets))
            # Calculate the number of tokens for this split
            split_number_tokens = torch.sum(torch.tensor([get_length(t) for t in filtered_targets]))
            valid_number_tokens.append(split_number_tokens)

    # Create a dictionary for each split containing the source, target, and other relevant information
    split_batches = []
    for i in range(len(valid_inputs)):
        split_batches.append({
            'source': valid_inputs[i],
            'target': valid_targets[i],
            'source_lengths': batch_dict['source_lengths'],
            'target_lengths': batch_dict['target_lengths'],
            'number_tokens': valid_number_tokens[i]  # Include number_tokens in each split
        })

    return split_batches
