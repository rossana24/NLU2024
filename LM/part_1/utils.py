# Add functions or classes used for data loading and preprocessing


import torch.utils.data as data
import torch
from functools import partial
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from collections import Counter


def get_data_loaders(args):
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

    """
    After padding with <pad>:
        Padded sentence 1: "This is a short sentence<pad><pad><pad><pad><pad>" (length: 10)
        Padded sentence 2: "This is a longer sentence<pad><pad><pad>" (length: 10)
    """
    lang = Lang(train_raw, ["<pad>", "<eos>"])

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
     and adds a user-defined end-o
    f-sentence token to each line. Finally, it returns a list of these processed strings.
     * You are reading line by lien, and you remove any
     * leading or ending whitespaces at the end of the sentence
     * put token at the end of each sentence
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
         * corpus (list):  list of strings, where each string represents a sentence or sequence
         of words
         special_tokens (list): some special tokens
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


class PennTreeBank(data.Dataset):

    def __init__(self, corpus, lang):
        """
        Summary
        This is a class inheriting data.Dataset and it will be used and passed to
        the data loader
        Input:
         * corpus (list):  list of strings, where each string represents a sentence or sequence of words
         * lang (lang): This is a class used for processing managing vocab in NLP>
        """
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
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
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


def calculate_statistics(tokenized_data, dataset_names, eos_token="<eos>"):
    """
    Summary:
    This function calculates and displays various statistics about tokenized datasets, such as the total number of words,
    the average sentence length, standard deviation, minimum and maximum sentence lengths.

    Input:
     * tokenized_data (list of lists): List containing tokenized sentences for each dataset
     * dataset_names (list of str): List of names corresponding to each dataset
     * eos_token (str): End-of-sequence token to exclude from length calculations

    Output:
     * None: Prints statistics and saves plots to files
    """

    dataset_lens = []
    # Compute sentence lengths for each dataset
    sent_lens = {name: [len([word for word in sent if word != eos_token]) for sent in data] for name, data in
                 zip(dataset_names, tokenized_data)}

    # Create a directory for plots if it doesn't exist
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for name, lengths in sent_lens.items():
        avg_sent_len = round(sum(lengths) / len(lengths))
        std_sent_len = np.std(lengths)
        min_sent_len = round(min(lengths))
        max_sent_len = round(max(lengths))

        # Print statistics
        print(f'\n{name} Set Statistics:')
        print(f'Total number of words: {sum(lengths)}')
        print(f'Total number of sentences: {len(lengths)}')
        print(f'Average sentence length: {avg_sent_len}')
        print(f'Standard deviation: {round(std_sent_len, 2)}')
        print(f'Min sentence length: {min_sent_len}')
        print(f'Maximum sentence length: {max_sent_len}')

        dataset_lens.append(len(lengths))

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.histplot(lengths, kde=True)
        plt.title(f'Distribution of Sentence Lengths in {name} Set')
        plt.xlabel('Sentence Length')
        plt.ylabel('Count')

        # Save the plot to a file
        plot_path = os.path.join(plot_dir, f'{name}_sentence_length_distribution.png')
        plt.savefig(plot_path)
        plt.close()  # Close the figure to avoid display

        print(f'Saved plot to {plot_path}')

    # Calculate and display the percentage distribution of each dataset
    total_length = sum(dataset_lens)
    percentage = [round((x / total_length) * 100, 3) for x in dataset_lens]
    df = pd.DataFrame(percentage, index=dataset_names, columns=['Set (%)'])
    print(df)


def calculate_top_word_frequencies(tokenized_train, tokenized_valid, tokenized_test, eos_token="<eos>", top_n=10):
    """
    Summary:
    This function calculates and prints the top N most frequent words in tokenized datasets.
    It computes the frequencies of the top N words and displays them as percentages of the total words.

    Input:
     * tokenized_train (list of lists): List of tokenized sentences for the training dataset
     * tokenized_valid (list of lists): List of tokenized sentences for the validation dataset
     * tokenized_test (list of lists): List of tokenized sentences for the test dataset
     * eos_token (str): End-of-sequence token to exclude from frequency calculations
     * top_n (int): Number of top frequent words to retrieve

    Output:
     * tuple: Contains dictionaries with top N word frequencies for train, validation, and test sets
    """

    def get_top_n_word_percentages(tokenized_data, top_n, eos_token="<eos>"):
        """
        Summary:
        This helper function calculates the top N most frequent words in a list of tokenized sentences.
        It returns the frequencies of these words as percentages of the total word count.

        Input:
         * tokenized_data (list of lists): List of tokenized sentences
         * top_n (int): Number of top frequent words to retrieve
         * eos_token (str): End-of-sequence token to exclude from frequency calculations

        Output:
         * dict: Dictionary of top N words and their frequencies as percentages
        """
        all_words = [word for sentence in tokenized_data for word in sentence if word != eos_token]
        freq_dist = Counter(all_words)
        total_words = len(all_words)
        top_words = freq_dist.most_common(top_n)
        return {word: round((count / total_words) * 100, 3) for word, count in top_words}

    # Calculate top N word frequencies for train, validation, and test sets
    top_freq_train = get_top_n_word_percentages(tokenized_train, top_n, eos_token)
    top_freq_valid = get_top_n_word_percentages(tokenized_valid, top_n, eos_token)
    top_freq_test = get_top_n_word_percentages(tokenized_test, top_n, eos_token)

    # Create DataFrames for better visualization
    df_train = pd.DataFrame.from_dict(top_freq_train, orient='index', columns=['Frequency'])
    df_valid = pd.DataFrame.from_dict(top_freq_valid, orient='index', columns=['Frequency'])
    df_test = pd.DataFrame.from_dict(top_freq_test, orient='index', columns=['Frequency'])

    # Print DataFrames
    print("Top word frequencies in the training set:")
    print(df_train)
    print("\nTop word frequencies in the validation set:")
    print(df_valid)
    print("\nTop word frequencies in the test set:")
    print(df_test)

    return top_freq_train, top_freq_valid, top_freq_test
