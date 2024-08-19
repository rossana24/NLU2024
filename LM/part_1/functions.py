# Add all the other required functions needed to complete the exercise.

import torch
import math
import torch.nn as nn
import random


def train_loop(args, data, optimizer, criterion, model, device):
    """
    Summary:
    This function performs the training loop for the model. It iterates over the training data, computes the loss,
    back-propagates the gradients, and updates the model weights.

    Input:
     * args (object): Argument parser containing model and training hyperparameters
     * data (DataLoader): Dataloader containing the training samples
     * optimizer (Optimizer): Optimizer for updating the model weights
     * criterion (Loss function): Loss function to compute the loss
     * model (nn.Module): The model to be trained
     * device (torch.device): Device on which to perform training (CPU or GPU)

    Output:
     * sum(loss_array) / sum(number_of_tokens) (float): Average loss per token over the training data
    """

    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        source_input = sample['source'].to(device)
        target_input = sample['target'].to(device)
        source_lengths = sample['source_lengths'].to(device)

        if args.Model.model_type == 'baseline_LSTM':
            output, hidden = model(source_input, source_lengths)
        else:
            output = model(source_input)

        loss = criterion(output, target_input)  # Compute the loss
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])

        loss.backward()  # Compute the gradient, deleting the computational graph
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.Training.clip)  # Clip the gradient to avoid explosion
        optimizer.step()  # Update the weights

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(args, data, eval_criterion, model, device):
    """
    Summary:
    This function performs the evaluation loop for the model. It iterates over the evaluation data,
    computes the loss, and calculates the perplexity.

    Input:
     * args (object): Argument parser containing model and training hyperparameters
     * data (DataLoader): Dataloader containing the evaluation samples
     * eval_criterion (Loss function): Loss function for evaluation
     * model (nn.Module): The model to be evaluated
     * device (torch.device): Device on which to perform evaluation (CPU or GPU)

    Output:
     * ppl, loss_to_return (tuple): Perplexity and average loss per token over the evaluation data
    """

    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():  # Avoid the creation of computational graph
        for sample in data:
            source_input = sample['source'].to(device)
            target_input = sample['target'].to(device)
            source_lengths = sample['source_lengths'].to(device) if 'source_lengths' in sample else None

            if args.Model.model_type == 'baseline_LSTM' and source_lengths is not None:
                output, hidden = model(source_input, source_lengths)
            else:
                output = model(source_input)
            loss = eval_criterion(output, target_input)  # Compute the loss
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    # Calculate perplexity and average loss per token
    avg_loss = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(avg_loss)

    return ppl, avg_loss


def init_weights(mat):
    """
    Summary:
    This function initializes the weights of the model. For recurrent layers (GRU, LSTM, RNN),
    it applies Xavier initialization for input weights and orthogonal initialization for hidden weights.
    For linear layers, it applies uniform initialization. Biases are initialized to zero or a small constant.

    Input:
     * mat (nn.Module): The model containing the layers whose weights need to be initialized

    Output:
     * None
    """

    for m in mat.modules():
        if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul:(idx + 1) * mul])  # Xavier's initialization for input weights
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(
                            param[idx * mul:(idx + 1) * mul])  # Orthogonal initialization for hidden weights
                elif 'bias' in name:
                    param.data.fill_(0)  # Initialize bias to zero

        elif isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)  # Uniform initialization for linear layer weights
            if m.bias is not None:
                m.bias.data.fill_(0.01)  # Initialize bias to a small constant


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

    # Adjust topk if 'unk' is not allowed and topk is set to 1
    if not unk and topk == 1:
        topk = 2

    # Generate words until end-of-sequence token or max length is reached
    while out_word != lang.word2id["<eos>"] and len(sent) < max_len:
        input = [lang.word2id[w] for w in sent]  # Convert input words to their corresponding ids

        length = torch.tensor([len(input)])
        length = length.to(device)

        input = torch.tensor(input)
        input = torch.unsqueeze(input, 0)
        input = input.to(device)

        out, hidden = model(input, length)
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
