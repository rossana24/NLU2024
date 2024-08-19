# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LockedDropout(nn.Module):
    """
    Summary:
    Implements a locked dropout mechanism, which applies the same dropout mask at each time step.
    This helps prevent the network from relying too much on specific time steps and improves generalization.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):

        if not self.training or not dropout:
            return x

        if isinstance(x, nn.utils.rnn.PackedSequence):
            data, batch_sizes = x.data, x.batch_sizes
            mask = data.new_ones(data.size(0), data.size(1)).bernoulli_(1 - dropout).to(data.device)
            mask = mask / (1 - dropout)
            masked_data = data * mask
            return nn.utils.rnn.PackedSequence(masked_data, batch_sizes)

        else:
            mask = x.new_ones(x.size(0), x.size(2)).bernoulli_(1 - dropout).to(x.device)
            mask = mask / (1 - dropout)
            mask = mask.unsqueeze(1).expand_as(x)
            return x * mask


class LM_LSTM_proposed_model(nn.Module):
    """
    Summary:
    This class implements an advanced Long Short-Term Memory (LSTM) network for language modeling.
    It includes embedding layers, multiple LSTM layers with variational dropout (LockedDropout),
    and a final linear output layer. The model also supports weight tying and weight dropout regularization.
    """

    def __init__(self, emb_size, hidden_size, output_size, device,
                 weight_drop_locked_i=0.1,
                 weight_drop_locked_h=0.1,
                 weight_drop_locked_o=0.1,
                 emb_dropout=0.1, pad_index=0, n_layers=1,
                 tie_weights=False,
                 tbptt=False,
                 tbptt_config=None,
                 ):
        """
        Input:
         * emb_size (int): Size of the embeddings (dimensionality of the embedding space)
         * hidden_size (int): Size of the hidden layer (dimensionality of LSTM hidden states)
         * output_size (int): Size of the output layer (number of classes or vocabulary size)
         * device (torch.device): Device to run the model on (CPU or GPU)
         * pad_index (int): Index of the padding token in the embedding layer
         * weight_drop_locked_i (float): Dropout rate applied to input embeddings before feeding into LSTM
         * weight_drop_locked_h (float): Dropout rate applied to hidden states between LSTM layers
         * weight_drop_locked_o (float): Dropout rate applied to LSTM output before the final linear layer
         * emb_dropout (float): Dropout rate for the embedding layer (unused in this implementation)
         * n_layers (int): Number of LSTM layers in the model
         * tie_weights (bool): Whether to tie weights between the embedding layer and the output layer
        """

        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tie_weights = tie_weights
        self.pad_index = pad_index

        # For TBPTT
        self.tbptt = tbptt
        self.tbptt_config = tbptt_config

        # weight_drop_locked_i: applied after embedding the input sequence and before feeding it into the LSTM layers
        self.weight_drop_locked_i = weight_drop_locked_i

        # weight_drop_locked_h: applied after the output of each LSTM layer, except for the final layer
        self.weight_drop_locked_h = weight_drop_locked_h

        # weight_drop_locked_o: applied after the final LSTM layer output and before feeding it into the fully
        # connected layer
        self.weight_drop_locked_o = weight_drop_locked_o

        self.embedding = nn.Embedding(self.output_size, self.emb_size, padding_idx=pad_index)
        self.lstm = [nn.LSTM(input_size=emb_size if i == 0 else hidden_size,
                             hidden_size=hidden_size if i != n_layers - 1 else (
                                 emb_size if tie_weights else hidden_size),
                             num_layers=1,
                             dropout=0,
                             bidirectional=False,
                             batch_first=True) for i in range(n_layers)]
        self.lstm = nn.ModuleList(self.lstm)

        # Optional: Variational Dropout
        # ____ To use the variational dropout and apply  the same dropout mask at each time step
        self.locked_dropout = LockedDropout()

        # ____ Uncomment to use normal dropout
        #self.locked_dropout = nn.Dropout()
        #self.dropout_i = nn.Dropout(weight_drop_locked_i)
        #self.dropout_h = nn.Dropout(weight_drop_locked_h)
        #self.dropout_o = nn.Dropout(weight_drop_locked_o)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Optional: Weight tying
        if tie_weights:
            self.fc.weight = self.embedding.weight
        self.init_weights()

    def forward(self, src, hidden, seq_lengths, batch_size, split_idx=None):

        embedding = self.embedding(src)

        # Optional: Variational Dropout
        embedding = self.locked_dropout(embedding, dropout=self.weight_drop_locked_i)
        # ____ Uncomment to use normal dropout
        #embedding = self.dropout_i(embedding)

        # For TBPTT
        if split_idx is not None and split_idx > 0:
            h_n, c_n = torch.stack(hidden[0], dim=0), torch.stack(hidden[1], dim=0)
            h_n = h_n[:, :src.shape[0], :].contiguous()
            c_n = c_n[:, :src.shape[0], :].contiguous()
            hidden = (h_n, c_n)

        if split_idx is not None:
            seq_lengths = seq_lengths.cpu().to(torch.int64)

        raw_output = pack_padded_sequence(embedding, seq_lengths, batch_first=True)
        new_hidden = []
        raw_outputs = []
        outputs = []

        # Ensure hidden list length matches the number of LSTM layers
        assert len(
            hidden) == self.n_layers, f"Hidden state list length {len(hidden)} does not match number of LSTM layers {self.n_layers}"

        for l, rnn in enumerate(self.lstm):
            current_input = raw_output
            current_batch = src.size(0)

            # Adjust hidden states for last batch size mismatch
            if current_batch < batch_size:
                hidden = [(h[0][:, :current_batch, :].contiguous(), h[1][:, :current_batch, :].contiguous())
                          for h in hidden]

            raw_output, new_h = rnn(current_input, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)

            # Apply variational dropout to intermediate LSTM outputs
            if l != self.n_layers - 1:
                # Optional: Variational Dropout
                # Comment the lines below to disable Variational Dropout
                raw_output = self.locked_dropout(raw_output, self.weight_drop_locked_h)
                outputs.append(raw_output)

                # ____ Uncomment to use normal dropout
                #raw_output, input_sizes = pad_packed_sequence(raw_output, batch_first=True)
                #raw_output = self.dropout_h(raw_output)
                #raw_output = pack_padded_sequence(raw_output, input_sizes, batch_first=True)
                #outputs.append(raw_output)

        hidden = new_hidden
        raw_output, input_sizes = pad_packed_sequence(raw_output, batch_first=True)

        # Optional: Variational Dropout
        # Comment the lines below to disable Variational Dropout
        raw_output = self.locked_dropout(raw_output, self.weight_drop_locked_o)

        # ____ Uncomment to use normal dropout
        #raw_output = self.dropout_o(raw_output)

        outputs.append(raw_output)
        prediction = self.fc(raw_output)
        return prediction.permute(0, 2, 1), hidden

    def tbptt_forward_wrapper(self, inputs, targets, lengths, n_tokens, optimizer, batch_size, criterion, clip):
        hiddens = self.init_hidden(inputs[0].shape[0])  # Initialize hidden states correctly

        batch_loss = 0.0
        total_tokens = 0

        pad_value = 0

        for split_idx, (inps, tars, tokens) in enumerate(zip(inputs, targets, n_tokens)):

            if split_idx > 0:
                hiddens = self.detach_hidden(hiddens)  # Detach hidden states

            lengths = torch.sum(inps.ne(pad_value), dim=1)

            outputs, hiddens = self(inps, hiddens, lengths, batch_size, split_idx)

            optimizer.zero_grad()  # Zero gradients for training

            loss = criterion(outputs, tars)

            # Multiply loss by the number of tokens in this split
            batch_loss += loss.item() * tokens
            total_tokens += tokens

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()

        # Normalize loss by the total number of tokens
        batch_loss /= total_tokens
        return batch_loss, None

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        device = next(self.parameters()).device
        return [(weight.new(1, batch, self.hidden_size if i != self.n_layers - 1 else
        (self.emb_size if self.tie_weights else self.hidden_size)).zero_().to(device),
                 weight.new(1, batch, self.hidden_size if i != self.n_layers - 1 else
                 (self.emb_size if self.tie_weights else self.hidden_size)).zero_().to(device))
                for i in range(self.n_layers)]

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hidden_size)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i, lstm_layer in enumerate(self.lstm):
            for name, param in lstm_layer.named_parameters():
                if 'weight_ih' in name:
                    param.data.uniform_(-init_range_other, init_range_other)
                elif 'weight_hh' in name:
                    param.data.uniform_(-init_range_other, init_range_other)

    def detach_hidden(self, hidden):
        detached_hidden = []
        for hidden_level in hidden:
            hidden, cell = hidden_level
            hidden = hidden.detach()
            cell = cell.detach()
            detached_hidden.append((hidden, cell))
        return detached_hidden

    def flatten_parameters(self):
        for rnn in self.lstm:
            rnn.flatten_parameters()
