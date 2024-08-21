# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from conll import evaluate
from sklearn.metrics import classification_report


class ModelIASBaseline(nn.Module):
    """
     Summary:
     This class defines a baseline neural network model for intent and slot prediction using a bidirectional LSTM.
     It includes embedding, LSTM, and linear layers, with added dropout for regularization.

     Input:
      * hidden_size (int): Number of features in the hidden state of the LSTM
      * out_slot (int): Number of slot labels (output size for slot filling)
      * output_intent (int): Number of intent labels (output size for intent classification)
      * emb_size (int): Size of the word embeddings
      * vocab_len (int): Size of the vocabulary
      * dropout_lstm (float): Dropout probability for the LSTM layer
      * dropout_linear (float): Dropout probability for the linear layers
      * dropout_emb (float): Dropout probability for the embedding layer
      * bidirectional (bool): Whether the LSTM is bidirectional or not
      * n_layer (int): Number of recurrent layers in the LSTM
      * pad_index (int): Index of the padding token in the vocabulary

     Output:
      * slots (Tensor): Logits for slot filling, shaped as (batch_size, classes, seq_len)
      * intent (Tensor): Logits for intent classification, shaped as (batch_size, output_intent)
    """

    def __init__(self, hidden_size, out_slot, output_intent, emb_size, vocab_len, dropout_lstm, dropout_linear,
                 dropout_emb, bidirectional=True, n_layer=1, pad_index=0):
        super(ModelIASBaseline, self).__init__()

        # Determine the input size of the linear layer based on bidirectionality
        linear_input_size = hidden_size * 2 if bidirectional else hidden_size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # Optional: Add Bidirectionality
        self.utt_encoder = nn.LSTM(emb_size, hidden_size, n_layer, dropout=dropout_lstm, bidirectional=bidirectional,
                                   batch_first=True)

        self.slot_out = nn.Linear(linear_input_size, out_slot)
        self.intent_out = nn.Linear(hidden_size, output_intent)

        # Optional: Adding dropout layer
        self.dropout_linear = nn.Dropout(dropout_linear)
        self.dropout_emb = nn.Dropout(dropout_emb)

    def forward(self, utterance, seq_lengths):
        utt_emb = self.dropout_emb(self.embedding(utterance))

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1, :, :]
        # Compute slot logits
        slots = self.dropout_linear(self.slot_out(utt_encoded))
        # Compute intent logits
        intent = self.dropout_linear(self.intent_out(last_hidden))

        slots = slots.permute(0, 2, 1)  # slots.size() = (batch_size, classes, seq_len)

        return slots, intent


PAD_TOKEN = 0


def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class NLUModel(pl.LightningModule):

    def __init__(self, args, lang):
        super(NLUModel, self).__init__()
        self.args = args
        self.lang = lang
        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)
        vocab_len = len(lang.word2id)
        self.model = ModelIASBaseline(emb_size=args.Model.emb_size,
                                      hidden_size=args.Model.hid_size,
                                      dropout_emb=args.Model.dropout_emb,
                                      dropout_linear=args.Model.dropout_linear,
                                      dropout_lstm=args.Model.dropout_lstm,
                                      bidirectional=args.Model.bidirectional,
                                      n_layer=args.Model.n_layers,
                                      out_slot=out_slot,
                                      output_intent=out_int,
                                      vocab_len=vocab_len,
                                      pad_index=PAD_TOKEN)
        self.model.apply(init_weights)
        self.criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.criterion_intents = nn.CrossEntropyLoss()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, utterances, slots_len):
        return self.model(utterances, slots_len)

    def training_step(self, batch, batch_idx):
        utterances = batch['utterances'].to(self.device)
        intents = batch['intents'].to(self.device)
        y_slots = batch['y_slots'].to(self.device)

        # Perform a forward pass to get predictions for slots and intents
        slots, intent = self(utterances, batch['slots_len'])

        # Calculate loss for intents and slots and store the computed losses
        loss_intent = self.criterion_intents(intent, intents)
        loss_slot = self.criterion_slots(slots, y_slots)
        loss = loss_intent + loss_slot
        self.train_step_outputs.append({"loss": loss, "loss_intent": loss_intent, "loss_slot": loss_slot})

        return loss

    def evaluation_step(self, batch, batch_idx, stage="val"):
        # Shared logic for validation and test steps

        utterances = batch['utterances'].to(self.device)
        intents = batch['intents'].to(self.device)  # ground truth
        y_slots = batch['y_slots'].to(self.device)

        # Perform a forward pass to get predictions
        slots, intent = self(utterances, batch['slots_len'])

        # Calculate loss for intents and slots
        loss_intent = self.criterion_intents(intent, intents)
        loss_slot = self.criterion_slots(slots, y_slots)
        loss = loss_intent + loss_slot

        # Convert predicted intents and ground truth intents to their string representations
        ref_intents = [self.lang.id2intent[x] for x in intents.tolist()]
        hyp_intents = [self.lang.id2intent[x] for x in
                       torch.argmax(intent, dim=1).tolist()]
        ref_slots = []
        hyp_slots = []
        output_slots = torch.argmax(slots, dim=1)

        # Iterate over each sequence in the batch
        for id_seq, seq in enumerate(output_slots):
            length = batch['slots_len'].tolist()[id_seq]  # Actual length of the current sequence
            utt_ids = batch['utterance'][id_seq][:length].tolist()  # Utterance IDs up to the actual length
            gt_ids = batch['y_slots'][id_seq].tolist()  # Ground truth slot IDs

            # Convert ground truth slots and utterances back to their string representations
            gt_slots = [self.lang.id2slot[elem] for elem in gt_ids[:length]]
            utterance = [self.lang.id2word[elem] for elem in utt_ids]

            # Convert predicted slot IDs back to their string representations
            to_decode = seq[:length].tolist()

            # Pair each word in the utterance with its corresponding predicted and ground truth slot labels
            ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
            hyp_slots.append([(utterance[id_el], self.lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

        # Return a dictionary with evaluation results
        return {
            f'{stage}_loss': loss,
            f'{stage}_loss_intent': loss_intent,
            f'{stage}_loss_slot': loss_slot,
            'ref_intents': ref_intents,
            'hyp_intents': hyp_intents,
            'ref_slots': ref_slots,
            'hyp_slots': hyp_slots
        }

    def validation_step(self, batch, batch_idx):
        outputs = self.evaluation_step(batch, batch_idx, stage="val")
        self.validation_step_outputs.append(outputs)
        return outputs['val_loss']

    def test_step(self, batch, batch_idx):
        outputs = self.evaluation_step(batch, batch_idx, stage="test")
        self.test_step_outputs.append(outputs)
        return outputs['test_loss']

    def on_train_epoch_end(self):
        # At the end of each training epoch, calculate the average losses

        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        avg_loss_intent = torch.stack([x['loss_intent'] for x in self.train_step_outputs]).mean()
        avg_loss_slot = torch.stack([x['loss_slot'] for x in self.train_step_outputs]).mean()

        self.log('train/loss_total', avg_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_intent', avg_loss_intent.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_slot', avg_loss_slot.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_step_outputs.clear()

    def on_evaluation_epoch_end(self, outputs, stage="val"):
        # At the end of each evaluation epoch (validation or test), calculate the average losses and other metrics

        avg_loss = torch.stack([x[f'{stage}_loss'] for x in outputs]).mean()
        avg_loss_intent = torch.stack([x[f'{stage}_loss_intent'] for x in outputs]).mean()
        avg_loss_slot = torch.stack([x[f'{stage}_loss_slot'] for x in outputs]).mean()

        # Flatten the lists of reference and hypothesis intents/slots
        ref_intents = sum([x['ref_intents'] for x in outputs], [])
        hyp_intents = sum([x['hyp_intents'] for x in outputs], [])
        ref_slots = sum([x['ref_slots'] for x in outputs], [])
        hyp_slots = sum([x['hyp_slots'] for x in outputs], [])

        len_of_len_ref_slots = sum([len(ref_slot) for ref_slot in ref_slots])
        len_of_len_hyp_slots = sum([len(hyp_slot) for hyp_slot in hyp_slots])
        assert len_of_len_ref_slots == len_of_len_hyp_slots, "number of slots is not the same"

        try:
            # Evaluate the slot predictions using custom evaluation metrics
            results = evaluate(ref_slots, hyp_slots)
        except Exception as ex:
            print(f"An error occurred and these are the len {len(ref_slots)} and {len(hyp_slots)}")
            print("Warning:", ex)
            ref_s = set([x[1] for x in ref_slots])
            hyp_s = set([x[1] for x in hyp_slots])
            print(hyp_s.difference(ref_s))
            results = {"total": {"f": 0}}

        # Generate a classification report for intent predictions
        report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)

        self.log(f'{stage}/loss', avg_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/loss_intent', avg_loss_intent.item(), on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(f'{stage}/loss_slot', avg_loss_slot.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/f1', results['total']['f'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}/acc', report_intent['accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end(self.validation_step_outputs, stage="val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(self.test_step_outputs, stage="test")
        self.test_step_outputs.clear()

    def on_save_checkpoint(self, checkpoint):
        print("Saving checkpoint...")
        checkpoint["word2id"] = self.lang.word2id
        checkpoint["slot2id"] = self.lang.slot2id
        checkpoint["intent2id"] = self.lang.intent2id
        checkpoint["args"] = self.args
        return checkpoint

    def configure_optimizers(self):
        if self.args.Training.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.args.Training.lr,
                                  weight_decay=self.args.Training.weight_decay)
        elif self.args.Training.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.args.Training.lr,
                                   weight_decay=self.args.Training.weight_decay)
        elif self.args.Training.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.args.Training.lr,
                                    weight_decay=self.args.Training.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.args.Training.optimizer}")
        return optimizer
