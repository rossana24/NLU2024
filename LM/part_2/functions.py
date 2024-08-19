# Add all the other required functions needed to complete the exercise.

import torch
import random
from utils import tbptt_split_batch


def train_loop(data, optimizer, criterion, model, device, batch_size, clip=5):
    """
    Summary:
    This function performs the training loop for the model. It iterates over the training data, computes the loss,
    back-propagates the gradients, and updates the model weights.

    Input:
     * data (DataLoader): Dataloader containing the training samples
     * optimizer (Optimizer): Optimizer for updating the model weights
     * criterion (Loss function): Loss function to compute the loss
     * model (nn.Module): The model to be trained
     * device (torch.device): Device on which to perform training (CPU or GPU)
     * batch_size (int): Size of the batches used in training
     * clip (float): Gradient clipping value to prevent exploding gradients

    Output:
     * float: Average loss per token over the training data
    """

    # Set the model to training mode
    model.train()
    model.to(device)
    loss_array = []
    number_of_tokens = []
    hidden = model.init_hidden(batch_size)

    for i, sample in enumerate(data):

        # Check if the model is using Truncated Backpropagation Through Time
        if model.tbptt:
            # Split the batch into smaller chunks according to the TBPTT configuration
            split_batches = tbptt_split_batch(sample, model.tbptt_config.mu,
                                              model.tbptt_config.std, model.tbptt_config.p,
                                              model.pad_index)

            inputs = [split_batch['source'].to(device) for split_batch in split_batches]
            targets = [split_batch['target'].to(device) for split_batch in split_batches]
            lengths = [split_batch['source_lengths'].to(device) for split_batch in split_batches]
            n_tokens = [split_batch['number_tokens'].to(device) for split_batch in split_batches]

            # Perform the forward and backward passes with TBPTT, accumulate the loss
            batch_loss, _ = model.tbptt_forward_wrapper(inputs, targets, lengths, n_tokens,
                                                        optimizer, batch_size,
                                                        criterion, clip)

            loss_array.append(batch_loss * sample["number_tokens"])
            number_of_tokens.append(sample['number_tokens'])

        else:
            # Standard training step
            optimizer.zero_grad()  # Zeroing the gradient
            source_input = sample['source'].to(device)
            target_input = sample['target'].to(device)

            # Forward pass
            output, hidden = model(source_input, hidden, sample['source_lengths'], batch_size)
            hidden = model.detach_hidden(hidden)

            loss = criterion(output, target_input)
            loss_array.append(loss.item() * sample["number_tokens"])
            number_of_tokens.append(sample["number_tokens"])
            loss.backward() # Backward pass: compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # clip the gradient to avoid explosion gradients
            optimizer.step()  # Update the weights

    # Compute the average loss per token
    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model, device, batch_size):
    """
    Summary:
    This function performs the evaluation loop for the model. It computes the loss and perplexity on the evaluation data.

    Input:
     * data (DataLoader): Dataloader containing the evaluation samples
     * eval_criterion (Loss function): Loss function to compute the evaluation loss
     * model (nn.Module): The model to be evaluated
     * device (torch.device): Device on which to perform evaluation (CPU or GPU)
     * batch_size (int): Size of the batches used in evaluation

    Output:
     * tuple: Perplexity (float) and average loss per token (float) over the evaluation data
    """

    # Set the model to evaluation mode
    model.eval()
    loss_array = []
    number_of_tokens = []
    hidden = model.init_hidden(batch_size)

    # No gradient computations needed during evaluation
    with torch.no_grad():
        for sample in data:
            source_input = sample['source'].to(device)
            target_input = sample['target'].to(device)

            output, hidden = model(source_input, hidden, sample['target_lengths'], batch_size)
            hidden = model.detach_hidden(hidden)

            loss = eval_criterion(output, target_input)
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    # Compute average loss per token
    average_loss = sum(loss_array) / sum(number_of_tokens)
    # Compute perplexity (exponential of the average loss)
    ppl = torch.exp(torch.tensor(average_loss)).item()

    return ppl, average_loss


def generate_sentence(model, s, lang, max_len=20, topk=1, unk=False, device='cuda:0'):
    """
    Summary:
    This function generates a sentence using the trained model. It takes an initial input sentence, processes it through
    the model, and appends the generated words until an end-of-sequence token is generated or the maximum length is reached.

    Input:
     * model (nn.Module): The trained model used for sentence generation
     * s (str): The initial input sentence to start the generation
     * lang (Language object): The language object containing word-to-id and id-to-word mappings
     * max_len (int): The maximum length of the generated sentence
     * topk (int): The number of top predictions to consider for each word (for sampling purposes)
     * unk (bool): Whether to allow 'unk' token in the generated sentence
     * device (torch.device): Device on which to perform sentence generation (CPU or GPU)

    Output:
     * sent (list): The generated sentence as a list of words
    """

    s = s.lower()
    sent = s.split(' ')
    out_word = ''

    hidden = model.init_hidden(1)

    # Adjust topk if 'unk' is not allowed and topk is set to 1
    if not unk and topk == 1:
        topk = 2

    # Generate words until end-of-sequence token or max length is reached
    while out_word != lang.word2id.get("<eos>", None) and len(sent) < max_len:
        input = [lang.word2id.get(w, lang.word2id.get('unk')) for w in sent]  # Convert input words to their corresponding ids

        length = torch.tensor([len(input)], dtype=torch.long).to(device)  # Ensure length is Long tensor
        length = length.cpu()  # Move length to CPU for pack_padded_sequence

        input = torch.tensor(input, dtype=torch.long).unsqueeze(0).to(device)  # Ensure input is Long tensor

        out, hidden = model(input, hidden, length, batch_size=1)
        out = out.reshape(-1, len(lang.word2id))
        out = torch.topk(out, topk, dim=-1).indices
        out = out[-1]  # Get the last token prediction

        if len(out) > 1:
            out = torch.squeeze(out, 0)

        out_w = random.choice(out)  # Randomly choose one of the top-k predictions

        # Ensure 'unk' token is not chosen if not allowed
        if not unk:
            while lang.id2word[out_w.item()] == 'unk':
                out_w = random.choice(out)

        out_word = lang.id2word[out_w.item()]
        sent.append(out_word)

    return sent

