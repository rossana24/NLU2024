# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LockedDropout(nn.Module):
    """
    Summary:
    This class implements a locked dropout mechanism, which applies the same dropout mask at each time step.
    This helps to prevent the network from relying too much on specific time steps and improves generalization.
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
            return mask * x


class LM_RNN(nn.Module):
    """
    Summary:
    This class implements a Recurrent Neural Network (RNN) for language modeling.
    It consists of an embedding layer, an RNN layer, and a linear output layer.
    """

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        """
        Input:
         * emb_size (int): Size of the embeddings
         * hidden_size (int): Size of the hidden layer
         * output_size (int): Size of the output layer (vocabulary size)
         * pad_index (int, optional): Index of the padding token (default: 0)
         * out_dropout (float, optional): Dropout rate for the output layer (default: 0.1)
         * emb_dropout (float, optional): Dropout rate for the embedding layer (default: 0.1)
         * n_layers (int, optional): Number of RNN layers (default: 1)
        """

        #super(LM_RNN, self).__init__()
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output


class LM_LSTM(nn.Module):
    """
    Summary:
    This class implements a Long Short-Term Memory (LSTM) network for language modeling.
    It consists of an embedding layer, an LSTM layer, dropout layers, and a linear output layer.
    """

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        """
        Input:
         * emb_size (int): Size of the embeddings
         * hidden_size (int): Size of the hidden layer
         * output_size (int): Size of the output layer
         * pad_index (int): Index of the padding token
         * out_dropout (float): Dropout rate for the LSTM layer
         * emb_dropout (float): Dropout rate for the embedding layer
         * n_layers (int): Number of LSTM layers
        """

        super().__init__()
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.emb_size, padding_idx=pad_index)
        
        # LSTM layer with dropout between layers
        self.lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            dropout=out_dropout, bidirectional=False, batch_first=True)

        self.dropout_emb = nn.Dropout(emb_dropout)
        self.dropout_lstm = nn.Dropout(out_dropout)

        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.init_weights()

    def forward(self, src, seq_lengths, hidden=None):
        # Optional: Additional Dropout
        # Comment the lines below to disable additional Dropout
        embedding = self.dropout_emb(self.embedding(src))
        # ____ Uncomment to use normal dropout
        #embedding = self.embedding(src)

        # Pack sequences to handle variable-length sequences efficiently
        packed_input = pack_padded_sequence(embedding, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.lstm(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        hidden = last_hidden[-1, :, :]

        # Optional: Additional Dropout
        # Comment the line below to disable additional Dropout
        output = self.dropout_lstm(output)

        prediction = self.fc(output)

        # Predictions reshaped to (Batch size, Length of sequence, Vocabulary size)
        return prediction.permute(0, 2, 1), hidden

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hidden_size)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.n_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_size,
                                                            self.hidden_size).uniform_(-init_range_other,
                                                                                       init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_size,
                                                            self.hidden_size).uniform_(-init_range_other,
                                                                                       init_range_other)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell

    def detach_hidden(self):
        hidden, cell = self
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

