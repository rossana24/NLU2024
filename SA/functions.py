import torch
import numpy as np

from evals import evaluate


def convert_to_bio(start_indices, end_indices, seq_len):
    """
    Convert predicted start and end positions into BIO-tagged list, considering the BERT input format.

    :param start_indices: List of predicted start positions of opinion targets.
    :param end_indices: List of predicted end positions of opinion targets.
    :param seq_len: Length of the input sequence.
    :return: List with BIO tags (O=0, B=1, I=2).
    """
    bio_labels = [0] * seq_len  # Initialize with 'O' (0)
    for tok_start_position, tok_end_position in zip(start_indices, end_indices):
        # Ensure valid start and end positions
        if tok_start_position >= 0 and tok_end_position <= seq_len - 1:
            start_position = tok_start_position  # Adjust for [CLS] token
            end_position = tok_end_position  # Adjust for [CLS] token

            # Mark the start position with 'B'
            bio_labels[start_position] = 1  # 'B'

            # Mark the positions within the span with 'I'
            if start_position < end_position:
                for idx in range(start_position + 1, end_position + 1):
                    bio_labels[idx] = 2  # 'I'

    return bio_labels


def magic(data):
    """
    Convert BIO tags to BIOES tags.

    :param data: List of BIO tags.
    :return: List of BIOES tags.
    """
    step_1 = bio2ot_ote(data)
    step_2 = ot2bieos_ote(step_1)
    return step_2


def bio2ot_ote(ote_tag_sequence):
    """
    Convert BIO tags to opinion targets (OT).

    :param ote_tag_sequence: List of BIO tags.
    :return: List of OT tags.
    """
    new_ote_sequence = []
    n_tags = len(ote_tag_sequence)
    for i in range(n_tags):
        ote_tag = ote_tag_sequence[i]
        if ote_tag == 'B' or ote_tag == 'I':
            new_ote_sequence.append('T')
        else:
            new_ote_sequence.append('I')
    return new_ote_sequence


def ot2bieos_ote(ote_tag_sequence):
    """
    Convert opinion targets (OT) to BIOES tags.

    :param ote_tag_sequence: List of OT tags.
    :return: List of BIOES tags.
    """
    n_tags = len(ote_tag_sequence)
    new_ote_sequence = []
    prev_ote_tag = '$$$'
    for i in range(n_tags):
        cur_ote_tag = ote_tag_sequence[i]
        if cur_ote_tag == 'O':
            new_ote_sequence.append('O')
        else:
            # cur_ote_tag is T
            if prev_ote_tag != cur_ote_tag:
                # prev_ote_tag is O, new_cur_tag can only be B or S
                if i == n_tags - 1:
                    new_ote_sequence.append('S')
                elif ote_tag_sequence[i + 1] == cur_ote_tag:
                    new_ote_sequence.append('B')
                elif ote_tag_sequence[i + 1] != cur_ote_tag:
                    new_ote_sequence.append('S')
                else:
                    raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
            else:
                # prev_tag is T, new_cur_tag can only be I or E
                if i == n_tags - 1:
                    new_ote_sequence.append('E')
                elif ote_tag_sequence[i + 1] == cur_ote_tag:
                    # next_tag is T
                    new_ote_sequence.append('I')
                elif ote_tag_sequence[i + 1] != cur_ote_tag:
                    # next_tag is O
                    new_ote_sequence.append('E')
                else:
                    raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
        prev_ote_tag = cur_ote_tag
    return new_ote_sequence


def logits_to_tags(start_logits, end_logits, gn_labels, threshold=3):
    """
    Convert start and end logits to BIO tags.

    :param start_logits: List of start logits for each token in the sequence.
    :param end_logits: List of end logits for each token in the sequence.
    :param gn_labels: List of ground truth BIO labels.
    :param threshold: Logit threshold for determining start and end positions.
    :return: Tuple containing converted predicted and ground truth BIOES tags.
    """
    number_to_tag = {0: 'O', 1: 'B', 2: 'I'}
    # Convert lists to numpy arrays
    start_logits = np.array(start_logits)
    end_logits = np.array(end_logits)

    # Identify Start and End Indices Using a Threshold
    start_indices = np.where(start_logits >= threshold)[0]
    end_indices = np.where(end_logits >= threshold)[0]
    span_starts, span_ends = [], []

    # Generate span indices and ensure that only valid spans are considered
    for start_index in start_indices:
        for end_index in end_indices:
            if start_index > end_index:
                continue  # Ignore invalid spans where the start index is after the end index
            if start_index >= len(start_logits):
                continue  # Ignore start indices that are out of bounds of the logits array
            if end_index >= len(start_logits):
                continue  # Ignores end indices that are out of bounds of the logits array

            span_starts.append(start_index)
            span_ends.append(end_index)

    # Account for single-token spans
    single_token_spans = set(start_indices).intersection(set(end_indices))
    for index in single_token_spans:
        span_starts.append(index)
        span_ends.append(index)

    predict_tag = convert_to_bio(span_starts, span_ends, len(start_logits))
    # Convert into BIO notation
    converted_predict_tag = [number_to_tag[number] for number in predict_tag]
    converted_gn_tag = [number_to_tag[number] for number in gn_labels]
    # Convert into BIOES notation
    converted_predict_tag_sbe = magic(converted_predict_tag)
    converted_gn_tag_sbe = magic(converted_gn_tag)

    return converted_predict_tag_sbe, converted_gn_tag_sbe


def run_valid_epoch(model, val_dataloader, device, threshold=0.5):
    """
    Run validation epoch to evaluate model performance.

    :param model: The model to evaluate.
    :param val_dataloader: DataLoader for validation data.
    :param device: The device (CPU or GPU) to run the evaluation.
    :param threshold: Threshold for converting logits to BIO tags.
    :return: None
    """
    model.eval()

    pred_tags_list = []
    gn_tags_list = []
    for step, batch in enumerate(val_dataloader):
        # Extract the necessary tensors from the batch
        input_ids = batch['input_ids'].to(device)
        input_mask = batch['attention_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        start_positions = batch['start_positions_ids'].to(device)
        end_positions = batch['end_positions_ids'].to(device)
        bio_labels = batch['bio_labels_ids']

        with torch.no_grad():
            results = model(input_ids, segment_ids, input_mask, start_positions, end_positions)

        for j, example_index in enumerate(batch['unique_ids']):
            len_of_padding = batch['len_input'][j].detach().cpu().item()
            start_logits = results['start_positions'][j].detach().cpu().tolist()[:len_of_padding]
            end_logits = results['end_positions'][j].detach().cpu().tolist()[:len_of_padding]
            gn_labels = bio_labels[j].detach().cpu().tolist()[:len_of_padding]

            # Convert start and end logist and ground truth labels into BIO format
            pred_tags, gn_tags = logits_to_tags(start_logits, end_logits, gn_labels, threshold=threshold)
            pred_tags_list.append(pred_tags)
            gn_tags_list.append(gn_tags)

    # Call the evaluate function
    print(len(val_dataloader.dataset))
    print(len(gn_tags_list))
    print(len(pred_tags_list))

    ote_scores = evaluate(gold_ot=gn_tags_list, pred_ot=pred_tags_list)

    # Print the results
    print("Precision: {:.4f}".format(ote_scores[0]))
    print("Recall: {:.4f}".format(ote_scores[1]))
    print("F1 Score: {:.4f}".format(ote_scores[2]))


def run_train_epoch(global_step, model, train_dataloader,
                    optimizer, device):
    """
    Execute one training epoch for the model.

    :param global_step: The current global training step (cumulative across all epochs).
    :param model: The model being trained.
    :param train_dataloader: DataLoader providing the training data.
    :param optimizer: The optimizer responsible for updating model parameters.
    :param device: The device on which to perform the computations (e.g., 'cuda' or 'cpu').
    """
    running_loss, count = 0.0, 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Extract the necessary tensors from the batch
        input_ids = batch['input_ids'].to(device)
        input_mask = batch['attention_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        start_positions = batch['start_positions_ids'].to(device)
        end_positions = batch['end_positions_ids'].to(device)

        # Perform forward pass to compute the model's predictions and loss.
        results = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
        loss = results['total_loss']

        loss.backward()  # Back-propagate the loss to compute gradients.
        running_loss += loss.item()  # Accumulate the loss for tracking.

        optimizer.step()
        model.zero_grad()
        global_step += 1
        count += 1

        print("step: {}, loss: {:.4f}".format(global_step, running_loss / count))


def prepare_optimizer(args, model):
    """
    Prepare the optimizer for model training, grouping parameters with and without weight decay.

    :param args: Arguments containing training hyperparameters like learning rate and weight decay.
    :param model: The model whose parameters will be optimized.
    :return: An AdamW optimizer configured with grouped parameters.
    """

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.Training.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.Training.lr,
                                  eps=args.Training.adam_epsilon)
    return optimizer


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """
    Save the model checkpoint to the specified path.

    :param model: The model to save.
    :param optimizer: The optimizer state to save.
    :param epoch: The current epoch number.
    :param checkpoint_path: The file path where the checkpoint will be saved.
    """
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")
