from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.utils.data as data


def read_absa_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return: List of dictionaries containing sentence and tags.
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')

            ts_tags = []  # tag sequence for targeted sentiment
            ote_tags = []  # tag sequence for opinion target extraction
            words = []

            # Processes each pair to identify words and their corresponding sentiment tags
            # valid sentiment tags are: O, T-POS, T-NEG, T-NEU
            for item in word_tag_pairs:
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                words.append(word.lower())

                # Map tags to their respective values
                if tag == 'O':
                    ote_tags.append('O')
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ote_tags.append('T')
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)

            record['words'] = words.copy()
            record['ote_raw_tags'] = ote_tags.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset


def ts2start_end(ts_tag_sequence):
    """
    Summary
    Identify the start and end indices of each sentiment span in the tag sequence.

    Input
     * ts_tag_sequence: List of tags for sentiment analysis (e.g., ['O', 'T-POS', 'T-NEG'])
    Output:
     *  starts and ends (lists): starts[i] corresponds to the starting index of the i-th sentiment span, ends[i]
        corresponds to the ending index of the i-th sentiment span.
    """

    starts, ends = [], []
    n_tag = len(ts_tag_sequence)
    prev_pos, prev_sentiment = '$$$', '$$$'  # Initialize previous position and sentiment
    tag_on = False  # Flag to indicate if a sentiment span is active

    # iterates over each tag in the ts_tag_sequence
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]

        if cur_ts_tag != 'O':
            # if it is not O
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
        else:
            cur_pos, cur_sentiment = 'O', '$$$'
        assert cur_pos == 'O' or cur_pos == 'T'

        # Determine if current tag starts or ends a sentiment span
        if cur_pos == 'T':
            if prev_pos != 'T':
                starts.append(i)  # New sentiment span starts
                tag_on = True
            else:
                if cur_sentiment != prev_sentiment:
                    ends.append(i - 1)  # End of previous sentiment span
                    starts.append(i)  # Start of new sentiment span
                    tag_on = True
        else:
            if prev_pos == 'T':
                ends.append(i - 1)  # End of sentiment span
                tag_on = False

        prev_pos = cur_pos
        prev_sentiment = cur_sentiment

    if tag_on:
        ends.append(n_tag - 1)  # End of the last sentiment span if still active

    assert len(starts) == len(ends), (len(starts), len(ends), ts_tag_sequence)

    return starts, ends


def pos2term(words, starts, ends):
    """
    Summary:
    Extract text spans from the list of words based on the start and end indices.
    """

    term_texts = []
    for start, end in zip(starts, ends):
        term_texts.append(' '.join(words[start:end + 1]))
    return term_texts


class SemEvalExample(object):
    def __init__(self,
                 example_id,
                 sent_tokens,
                 term_texts=None,
                 start_positions=None,
                 end_positions=None):
        self.example_id = example_id
        self.sent_tokens = sent_tokens
        self.term_texts = term_texts
        self.start_positions = start_positions
        self.end_positions = end_positions


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_positions=None,
                 end_positions=None,
                 start_indexes=None,
                 end_indexes=None,
                 bio_labels=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.bio_labels = bio_labels


def convert_absa_data(dataset, verbose_logging=False):
    """
    Summary
    Convert the dataset into examples for training or evaluation.

    Input:
     * dataset: List of records with 'sentence', 'words', 'ote_raw_tags', 'ts_raw_tags'
     * verbose_logging: Whether to print detailed logs
    Output:
     * List of SemEvalExample objects
    """
    examples = []
    n_records = len(dataset)

    for i in range(n_records):
        words = dataset[i]['words']  # List of words in the sentence
        ts_tags = dataset[i]['ts_raw_tags']  # List of sentiment tags (O, Pos, Neut, Neg)

        # Convert tag sequence to start and end positions of spans
        starts, ends = ts2start_end(ts_tags)
        term_texts = pos2term(words, starts, ends)

        if term_texts != []:  # Ensure non-empty term texts
            assert len(term_texts) == len(starts)
            # Create a SemEvalExample object for this example
            example = SemEvalExample(str(i), words, term_texts, starts, ends)
            examples.append(example)

            if i < 50 and verbose_logging:
                print(example)

    print("Convert %s examples" % len(examples))
    return examples


class CollateFn:
    def __init__(self, word_pad_token, general_token):
        self.word_pad_token = word_pad_token
        self.general_token = general_token

    def __call__(self, batch):
        def merge(sequences, pad_token):
            lengths = [len(seq) for seq in sequences]
            max_len = max(lengths)
            padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq
            return padded_seqs, lengths

        batch.sort(key=lambda x: len(x['input_ids']), reverse=True)

        # Creates a new dictionary new_item where each key corresponds to a list of values from all samples.
        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = [d[key] for d in batch]

        # Merge and pad each field in the batch
        unique_ids, _ = merge(new_batch['unique_ids'], 0)
        input_ids, y_lengths = merge(new_batch['input_ids'], self.word_pad_token)
        attention_ids, _ = merge(new_batch['attention_ids'], self.general_token)
        segment_ids, _ = merge(new_batch['segment_ids'], self.general_token)
        start_positions, _ = merge(new_batch['start_positions_ids'], self.general_token)
        end_positions, _ = merge(new_batch['end_positions_ids'], self.general_token)
        bio_labels, _ = merge(new_batch['bio_labels_ids'], self.general_token)

        new_batch['unique_ids'] = unique_ids
        new_batch["input_ids"] = input_ids
        new_batch["attention_ids"] = attention_ids
        new_batch["segment_ids"] = segment_ids
        new_batch["start_positions_ids"] = start_positions
        new_batch["end_positions_ids"] = end_positions
        new_batch["bio_labels_ids"] = bio_labels
        new_batch["len_input"] = torch.LongTensor(y_lengths)

        return new_batch


class Slots(data.Dataset):
    def __init__(self, dataset_examples, tokenizer):
        self.tokenizer = tokenizer
        self.examples = dataset_examples
        self.final_features = self.convert_text_to_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feature = self.final_features[idx]
        input_unique_id = torch.tensor([feature.unique_id], dtype=torch.long)
        all_input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        all_start_positions = torch.tensor(feature.start_positions, dtype=torch.long)
        all_end_positions = torch.tensor(feature.end_positions, dtype=torch.long)
        all_bio_labels = torch.tensor(feature.bio_labels, dtype=torch.long)

        sample = {
            "unique_ids": input_unique_id,
            "input_ids": all_input_ids,
            "attention_ids": all_input_mask,
            "segment_ids": all_segment_ids,
            "start_positions_ids": all_start_positions,
            "end_positions_ids": all_end_positions,
            "bio_labels_ids": all_bio_labels
        }
        return sample

    def convert_text_to_features(self):
        """
        Summary:
        Converts raw text data into model input features suitable for training or evaluation.

        This function tokenizes the text, maps tokens to IDs, creates attention masks,
        segment IDs, and generates positions for start and end of terms. It also handles
        padding to ensure all sequences in the batch are of the same length.

        Output:
         * List of SemEvalFeatures objects, each containing tokenized data and other features
        """
        # Sentences can have more than one span, the longest span is extracted
        max_term_num = max([len(example.term_texts) for (example_index, example) in enumerate(self.examples)])
        max_sent_length = 0
        unique_id = 1000000000  # A unique identifier for each example (starting from a large arbitrary number).
        features = []

        for (example_index, example) in enumerate(self.examples):
            tok_to_orig_index = []  # Maps tokenized words back to their original indices in the sentence.
            orig_to_tok_index = []  # Maps original words to their tokenized indices.
            all_doc_tokens = []  # Stores the tokenized version of the entire sentence.

            # Tokenize the sentence
            for (i, token) in enumerate(example.sent_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))  # Index of original token in tokenized list
                sub_tokens = self.tokenizer.tokenize(token)
                if not sub_tokens:
                    sub_tokens = '[UNK]'  # Handle unknown tokens by assigning [UNK]
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)  # Map each sub-token back to original token
                    all_doc_tokens.append(sub_token)

            ###  EXAMPLE of the tokenization step
            # The sentence ["The", "battery", "life", "is", "amazing"]
            # gets tokenized into ["The", "bat", "##tery", "life", "is", "amazing"].
            # orig_to_tok_index = [0, 1, 3, 4, 5] you see in here that you are missing the 2

            if len(all_doc_tokens) > max_sent_length:
                max_sent_length = len(all_doc_tokens)  # Update maximum sentence length

            # Determine token positions for start and end of te
            tok_start_positions = []
            tok_end_positions = []

            for start_position, end_position in zip(example.start_positions, example.end_positions):
                tok_start_position = orig_to_tok_index[start_position]
                if end_position < len(example.sent_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position + 1] - 1
                else:
                    # If end_position is the last token, end_position + 1 would be out of bounds,
                    # so the code uses else to map it to the last sub-word token in all_doc_tokens
                    # (i.e., len(all_doc_tokens) - 1).
                    tok_end_position = len(all_doc_tokens) - 1

                tok_start_positions.append(tok_start_position)
                tok_end_positions.append(tok_end_position)

            # Prepare the tokens for model input
            tokens = ["[CLS]"]  # CLS token for BERT-like models
            segment_ids = [0]  # Segment IDs (0 for single sentence input)
            token_to_orig_map = {}

            for index, token in enumerate(all_doc_tokens):
                token_to_orig_map[len(tokens)] = tok_to_orig_index[index]
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            # Convert tokens to IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)  # Attention mask: 1 for real tokens

            # Initialize position vectors
            start_positions = [0] * len(input_ids)
            end_positions = [0] * len(input_ids)
            bio_labels = [0] * len(input_ids)  # BIO labels (0: non-term, 1: start of term, 2: end of term)

            # Assign positions for start and end and convert them into BIO labels
            start_indexes, end_indexes = [], []
            for tok_start_position, tok_end_position in zip(tok_start_positions, tok_end_positions):
                if tok_start_position >= 0 and tok_end_position <= len(input_ids) - 1:
                    start_position = tok_start_position + 1  # [CLS]
                    end_position = tok_end_position + 1  # [CLS]
                    start_positions[start_position] = 1
                    end_positions[end_position] = 1
                    start_indexes.append(start_position)
                    end_indexes.append(end_position)
                    bio_labels[start_position] = 1  # 'B'
                    if start_position < end_position:
                        for idx in range(start_position + 1, end_position + 1):
                            bio_labels[idx] = 2  # 'I'

            while len(start_indexes) < max_term_num:
                start_indexes.append(0)
                end_indexes.append(0)

            assert len(start_indexes) == max_term_num
            assert len(end_indexes) == max_term_num

            # Create and store the feature object
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    start_indexes=start_indexes,
                    end_indexes=end_indexes,
                    bio_labels=bio_labels)
            )
            unique_id += 1  # Increment unique ID for the next example

        return features


def get_dataloaders(config, train: bool = False):
    if train:
        tag = "Train dataset"
        data_set = read_absa_data(config.Dataset.train_path)
    else:
        tag = "Test dataset"
        data_set = read_absa_data(config.Dataset.test_path)
    data_examples = convert_absa_data(dataset=data_set)
    max_length = max([len(item.sent_tokens) for item in data_examples])
    print(f"The max length of tokens in sentence in {tag} is {max_length}")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = Slots(data_examples, bert_tokenizer)
    collate_fn = CollateFn(
        word_pad_token=bert_tokenizer.pad_token_id,
        general_token=0)
    if train:
        data_sampler = RandomSampler(dataset)
        batch_size = config.Dataset.batch_size_train
    else:
        data_sampler = SequentialSampler(dataset)
        batch_size = config.Dataset.batch_size_test
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size,
                            collate_fn=collate_fn)
    return data_examples, dataloader
