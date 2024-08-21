import pytorch_lightning as pl
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from sklearn.metrics import classification_report
from conll import evaluate
import torch


class IntentClassifier(nn.Module):
    """
    Summary:
    This class defines a simple intent classifier using a linear layer and dropout for regularization.

    Input:
    * input_dim (int): Dimension of the input features
    * num_intent_labels (int): Number of intent labels (output size for intent classification)
    * dropout_rate (float): Dropout probability for the linear layer

    Output:
    * x (Tensor): Logits for intent classification, shaped as (batch_size, num_intent_labels)
    """

    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class IntentClassifier_aggregate(nn.Module):
    """
    Summary:
    This class defines an intent classifier that aggregates features from both the BERT pooled output
    and slot logits using a linear layer with dropout for regularization.

    Input:
    * hidden_size (int): Size of the BERT hidden state
    * num_slot_labels (int): Number of slot labels (output size for slot filling)
    * num_intent_labels (int): Number of intent labels (output size for intent classification)
    * dropout_rate (float): Dropout probability for the linear layer

    Output:
    * combined_representation (Tensor): Logits for intent classification, shaped as (batch_size, num_intent_labels)
    """

    def __init__(self, hidden_size, num_slot_labels, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier_aggregate, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # Linear layer that combines BERT pooled output and slot logits
        self.linear = nn.Linear(hidden_size + num_slot_labels, num_intent_labels)

    def forward(self, pooled_output, slots_aggregated):
        pooled_output = self.dropout(pooled_output)
        slots_aggregated = self.dropout(slots_aggregated)
        # Concatenate the pooled output and slot logits
        combined_representation = torch.cat([pooled_output, slots_aggregated], dim=1)
        return self.linear(combined_representation)  # Compute logits for intent classification


class SlotClassifier(nn.Module):
    """
    Summary:
    This class defines a slot classifier using a linear layer and dropout for regularization.

    Input:
    * input_dim (int): Dimension of the input features
    * num_slot_labels (int): Number of slot labels (output size for slot filling)
    * dropout_rate (float): Dropout probability for the linear layer

    Output:
    * x (Tensor): Logits for slot filling, shaped as (batch_size, num_slot_labels)
    """

    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class BertModelSlotsIntent(BertPreTrainedModel):
    """
    Summary:
    This class defines a model for joint intent classification and slot filling using a pre-trained BERT model.
    It allows optional use of a 1D convolutional layer for slot filling and an aggregation step for intent classification.

    Input:
    * input_ids (Tensor): Tensor of input token IDs, shaped as (batch_size, seq_len)
    * attention_mask (Tensor): Tensor of attention masks, shaped as (batch_size, seq_len)
    * token_type_ids (Tensor): Tensor of token type IDs, shaped as (batch_size, seq_len)

    Output:
    * intent_logits (Tensor): Logits for intent classification, shaped as (batch_size, num_intent_labels)
    * slot_logits (Tensor): Logits for slot filling, shaped as (batch_size, seq_len, num_slot_labels)
    """

    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(BertModelSlotsIntent, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        # Optional: Uncomment to apply a 1D convolutional layer for slot logits
        self.conv1d = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=3,
                                padding=1)

        # Intent classifier using pooled output from BERT
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.Training.dropout)

        # Uncomment this line to use an intent classifier that aggregates slot logits
        # self.intent_classifier = IntentClassifier_aggregate(config.hidden_size, self.num_slot_labels,
        #                                                     self.num_intent_labels, args.Training.dropout)

        # Slot classifier using sequence output from BERT
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.Training.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Optional: Uncomment to apply a 1D convolutional layer for slot logits
        #sequence_output = sequence_output.permute(0, 2, 1)  # (batch_size, hidden_size, seq_length)
        #conv_output = self.conv1d(sequence_output)
        #conv_output = torch.relu(conv_output)  # Apply activation function
        #conv_output = conv_output.permute(0, 2, 1)  # Back to (batch_size, seq_length, hidden_size)
        #slot_logits = self.slot_classifier(conv_output)

        slot_logits = self.slot_classifier(sequence_output)

        # Uncomment this line to use an intent classifier that aggregates slot logits
        #slots_aggregated = slot_logits.max(dim=1)[0]  # Shape: [batch_size, num_slot_labels]
        #intent_logits = self.intent_classifier(pooled_output, slots_aggregated)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = slot_logits.permute(0, 2, 1)

        return intent_logits, slot_logits


contraction_map = {
    "i ' d": "i 'd",
    "o ' clock": "o'clock",
    "st .": "st.",
    "i ' m": "i 'm",
    "what ' s": "what 's",
    "' s": "'s",
    "i ' ll": "i 'll",
    "we ' re": "we 're",
    "don ' t": " don 't",
    "what ' re": "what 're",
    "doesn ' t": "doesn 't",
    "o ' hare": "o 'hare",
    "i ' ve": "i 've"
}


def custom_detokenize(text):
    for key, value in contraction_map.items():
        text = text.replace(key, value)
    return text


# LightningModule for training and evaluating the BERT model for intent classification and slot filling
class NluModelBert(pl.LightningModule):
    """
    Summary:
    This class wraps the BERT model for joint intent classification and slot filling into a PyTorch Lightning module.
    It handles training, evaluation, and optimization.

    Input:
    * args: Arguments containing model and training configurations
    * total_num_steps (int): Total number of training steps
    * warm_up_steps (int): Number of warm-up steps for the learning rate scheduler
    * lang_model: Pre-trained language model
    * tokenizer: Tokenizer corresponding to the pre-trained language model

    Output:
    * forward: Forward pass through the BERT model
    """

    def __init__(self, args, total_num_steps, warm_up_steps=0, lang_model=None, tokenizer=None):
        super(NluModelBert, self).__init__()
        self.args = args
        self.lang = lang_model
        self.intent_label_lst = lang_model.id2intent
        self.slot_label_lst = lang_model.id2slot
        self.pad_token_label_id = args.Model.pad_id
        self.total_steps = total_num_steps
        self.warm_up_steps = warm_up_steps
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-uncased', finetuning_task=args.Dataset.data_set_name)
        self.model = BertModelSlotsIntent.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
                                                          config=self.config,
                                                          args=args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
        self.criterion_slots = nn.CrossEntropyLoss(ignore_index=self.pad_token_label_id)
        self.criterion_intents = nn.CrossEntropyLoss()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        utterance = batch['utterance'].to(self.device)
        attention = batch["attention"].to(self.device)
        token_type_ids = batch["token_type"].to(self.device)
        intents = batch["intents"].to(self.device)
        slots = batch["slots"].to(self.device)

        inputs = {'input_ids': utterance,
                  'attention_mask': attention,
                  'token_type_ids': token_type_ids,
                  }

        intent_logits, slot_logits = self(**inputs)

        # Compute the cross-entropy loss for slot filling
        loss_slot = self.criterion_slots(slot_logits, slots)

        loss_intent = self.criterion_intents(intent_logits, intents)
        loss = loss_intent + self.args.Training.slot_loss_coef * loss_slot
        self.train_step_outputs.append({"loss": loss, "loss_intent": loss_intent, "loss_slot": loss_slot})
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # Log the learning rate
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])

    def evaluation_step(self, batch, batch_idx, stage="val"):
        utterances = batch['utterance'].to(self.device)
        attention = batch["attention"].to(self.device)
        token_type_ids = batch["token_type"].to(self.device)
        intents = batch["intents"].to(self.device)
        slots = batch["slots"].to(self.device)
        inputs = {'input_ids': utterances,
                  'attention_mask': attention,
                  'token_type_ids': token_type_ids,
                  }

        # Forward pass: obtain model predictions
        intent_logits, slot_logits = self(**inputs)

        # Compute the loss for intent classification nad slot filling
        loss_intent = self.criterion_intents(intent_logits, intents)
        loss_slot = self.criterion_slots(slot_logits, slots)
        loss = loss_intent + self.args.Training.slot_loss_coef * loss_slot

        # Convert reference intent IDs to labels
        ref_intents = [self.lang.id2intent[x] for x in intents.tolist()]
        # Convert predicted intent logits to labels
        hyp_intents = [self.lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]

        ref_slots = []
        hyp_slots = []

        output_slots = torch.argmax(slot_logits, dim=1)

        # Detokenize and process each sequence in the batch
        for id_seq, seq in enumerate(output_slots):
            length = batch['slots_len'].tolist()[id_seq]
            utt_ids = batch['utterance'][id_seq][:length].tolist()
            gt_ids = batch['slots'][id_seq][:length].tolist()

            # Convert token IDs to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(utt_ids)
            # Remove special tokens and replace [UNK] tokens
            utterance = [token for token in tokens if token not in ['[CLS]', '[SEP]']]
            utterance = ["unk" if token == '[UNK]' else token for token in utterance]
            utterance = custom_detokenize(self.tokenizer.convert_tokens_to_string(utterance)).split()

            gt_slots = []
            to_decode = []

            # Map ground truth and predicted slot IDs to slot labels
            for index, gt_id in enumerate(gt_ids):
                # Skip padding tokens inserted in the middle of the sentence after tokenization
                if gt_id != self.pad_token_label_id:
                    gt_slots.append(self.lang.id2slot[gt_id])
                    to_decode.append(self.lang.id2slot[seq[index].detach().cpu().item()])

            assert len(utterance) == len(
                to_decode), f"Length mismatch: utterance ({len(utterance)}) != to_decode ({len(to_decode)})"
            assert len(utterance) == len(
                gt_slots), f"Length mismatch: utterance ({len(utterance)}) != to_decode ({len(gt_slots)})"

            ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
            hyp_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(to_decode)])

        # Return the results including losses and predictions
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
        print(outputs)
        return outputs['test_loss']

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        avg_loss_intent = torch.stack([x['loss_intent'] for x in self.train_step_outputs]).mean()
        avg_loss_slot = torch.stack([x['loss_slot'] for x in self.train_step_outputs]).mean()
        self.log('train/loss_total', avg_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_intent', avg_loss_intent.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_slot', avg_loss_slot.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_step_outputs.clear()

    def on_evaluation_epoch_end(self, outputs, stage="val"):
        avg_loss = torch.stack([x[f'{stage}_loss'] for x in outputs]).mean()
        avg_loss_intent = torch.stack([x[f'{stage}_loss_intent'] for x in outputs]).mean()
        avg_loss_slot = torch.stack([x[f'{stage}_loss_slot'] for x in outputs]).mean()

        ref_intents = sum([x['ref_intents'] for x in outputs], [])
        hyp_intents = sum([x['hyp_intents'] for x in outputs], [])
        ref_slots = sum([x['ref_slots'] for x in outputs], [])
        hyp_slots = sum([x['hyp_slots'] for x in outputs], [])

        len_of_len_ref_slots = sum([len(ref_slot) for ref_slot in ref_slots])
        len_of_len_hyp_slots = sum([len(hyp_slot) for hyp_slot in hyp_slots])

        assert len_of_len_ref_slots == len_of_len_hyp_slots, "number of slots is not the same"

        try:
            results = evaluate(ref_slots, hyp_slots)
        except Exception as ex:
            print(f"An error occurred and these are the len {len(ref_slots)} and {len(hyp_slots)}")
            print("Warning:", ex)
            ref_s = set([x[1] for x in ref_slots])
            hyp_s = set([x[1] for x in hyp_slots])
            print(hyp_s.difference(ref_s))
            results = {"total": {"f": 0}}

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
        checkpoint["vocab"] = self.lang.vocab
        checkpoint["slot2id"] = self.lang.slot2id
        checkpoint["intent2id"] = self.lang.intent2id
        checkpoint["pad_token"] = self.lang.pad_token

        checkpoint["config"] = self.config
        checkpoint["args"] = self.args
        return checkpoint  # Let Lightning save the rest of the checkpoint

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.Training.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.Training.lr,
                                      eps=self.args.Training.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warm_up_steps,
                                                    num_training_steps=self.total_steps)

        return [optimizer], [scheduler]
