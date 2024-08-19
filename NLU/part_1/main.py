import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
import os
import json
from utils import get_data_loaders, Lang
from model import ModelIASBaseline, NLUModel
from conll import evaluate
from sklearn.metrics import classification_report
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Attempt to import wandb for experiment tracking
try:
    import wandb

    deactivate_wandb = 0
except ImportError:
    print("WANDB is not available please install")
    print("use command: pip3 install wandb")
    deactivate_wandb = 1

"""
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

        slots, intent = self(utterances, batch['slots_len'])
        # this self is just to call the forward pass

        loss_intent = self.criterion_intents(intent, intents)
        loss_slot = self.criterion_slots(slots, y_slots)
        loss = loss_intent + loss_slot
        self.train_step_outputs.append({"loss": loss, "loss_intent": loss_intent, "loss_slot": loss_slot})

        return loss

    def evaluation_step(self, batch, batch_idx, stage="val"):
        utterances = batch['utterances'].to(self.device)
        intents = batch['intents'].to(self.device) # ground truth
        y_slots = batch['y_slots'].to(self.device)
        slots, intent = self(utterances, batch['slots_len'])
        loss_intent = self.criterion_intents(intent, intents)
        loss_slot = self.criterion_slots(slots, y_slots)
        loss = loss_intent + loss_slot

        ref_intents = [self.lang.id2intent[x] for x in intents.tolist()] # groung truth as word
        hyp_intents = [self.lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()] # prediction intents as words
        ref_slots = []
        hyp_slots = []
        output_slots = torch.argmax(slots, dim=1)

        # la parte qui sotto copiata
        for id_seq, seq in enumerate(output_slots):
            # It gets the actual length of the utterance.
            # get the length of the specific sentence in the batch
            length = batch['slots_len'].tolist()[id_seq]
            # Retrieves the original utterance and ground truth slot labels
            # make sure you donot include the padding that why you do untill :length
            utt_ids = batch['utterance'][id_seq][:length].tolist()
            # why you donot have :length in here? he is doing in the next step
            gt_ids = batch['y_slots'][id_seq].tolist()
            #  Converts the predicted and ground truth slot labels to their string representations.
            gt_slots = [self.lang.id2slot[elem] for elem in gt_ids[:length]]
            # convert the word back to its orginal format back again
            utterance = [self.lang.id2word[elem] for elem in utt_ids]
            # make sure that you donot have anything more than until the length
            to_decode = seq[:length].tolist()
            # Pairs each word in the utterance with its corresponding predicted and ground truth slot label.
            # you are creating word then the prediction
            # this how it has to be done
            ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
            hyp_slots.append([(utterance[id_el], self.lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

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
        len_of_len_ref_slots = sum([len(ref_slot)for ref_slot in ref_slots])
        len_of_len_hyp_slots = sum([len(hyp_slot)for hyp_slot in hyp_slots])
        print(len_of_len_ref_slots)
        print(len_of_len_hyp_slots)
        assert len_of_len_ref_slots==len_of_len_hyp_slots, "number of slots is not the same"

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
        checkpoint["word2id"]=self.lang.word2id
        checkpoint["slot2id"]=self.lang.slot2id
        checkpoint["intent2id"]=self.lang.intent2id
        checkpoint["args"]=self.args
        return checkpoint  # Let Lightning save the rest of the checkpoint

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

"""


def main(args):
    """
    Main function to handle training and evaluation of the model.

    Args:
      config (omegaconf.dictconfig.DictConfig): Configuration object containing parameters for training and evaluation.
    """

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This code is running on: {device}")

    if args.Training.eval_path == "":

        # Define the run name and create the directory for saving checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_run_name = f'{timestamp}_{args.Training.run_name}_{args.Training.tag}'
        checkpoints_dir = os.path.join('bin', 'bin', new_run_name)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Save configuration parameters to a JSON file
        args_dict = OmegaConf.to_container(args, resolve=True)
        args_file_path = os.path.join(checkpoints_dir, 'parameters.json')
        with open(args_file_path, 'w') as args_file:
            json.dump(args_dict, args_file, indent=4)

        # Initialize wandb
        wandb_logger = WandbLogger(name=new_run_name, project=args.Training.project_name, config=args_dict,
                                   log_model=False)
        wandb_logger.experiment.config.update({'x_axis': 'epoch'})

        # Data preparation: Load data loaders, language model, and tokenizer
        train_loader, val_loader, test_loader, lang = get_data_loaders(args)

        print(" dataset size ")
        dataset_size_train = len(train_loader.dataset)
        print(f'Dataset size: {dataset_size_train}')
        print(f'dataloader size :{len(train_loader)}')
        batch_size_train = train_loader.batch_size
        print(f'Batch size: {batch_size_train}')
        print("end data loader for train")
        print("##############################################")
        dataset_size_val = len(val_loader.dataset)
        print(f'Dataset size: {dataset_size_val}')
        print(f'dataloader size :{len(val_loader)}')
        batch_size_dev = val_loader.batch_size
        print(f'Batch size: {batch_size_dev}')
        print("end data loader for validation")
        print("##############################################")
        dataset_size_test = len(test_loader.dataset)
        print(f'Dataset size: {dataset_size_test}')
        print(f'dataloader size :{len(test_loader)}')
        batch_size_test = test_loader.batch_size
        print(f'Batch size: {batch_size_test}')
        print("end data loader for validation")
        print("##############################################")

        # Define callbacks for checkpointing and early stopping
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename='best_checkpoint',
            save_top_k=1,
            monitor='val/f1',
            mode='max',
            save_last=False,  # Don't save the last epoch, only the best one
            save_weights_only=True,  # Save the entire model (including optimizer state, etc.)
        )
        early_stop_callback = EarlyStopping(
            monitor='val/f1',
            min_delta=0.00,
            patience=args.Training.patience,
            verbose=True,
            mode='max'
        )

        # Initialize model
        model = NLUModel(args, lang)

        # Setup PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=args.Training.num_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            log_every_n_steps=1  # None  # This disables per-step logging
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Test the model
        trainer.test(model, test_loader)

    else:
        # Load model and language model from checkpoint
        checkpoint_path = os.path.join(args.Training.eval_path, "best_checkpoint.ckpt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        load_lang = Lang.load(checkpoint)
        train_loader, val_loader, test_loader, lang = get_data_loaders(args, load_lang)

        load_args = checkpoint['args']

        # Initialize model from checkpoint
        model = NLUModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device, strict=True,
                                              lang=load_lang, args=load_args)

        # Validate weights match between loaded checkpoint and model
        checkpoint_state_dict = model.state_dict()
        original_state_dict = checkpoint['state_dict']
        for key in original_state_dict:
            original_state_dict[key] = original_state_dict[key].to(device)
            checkpoint_state_dict[key] = checkpoint_state_dict[key].to(device)

            if not torch.equal(original_state_dict[key], checkpoint_state_dict[key]):
                print(f"Mismatch found in layer: {key}")
                break
        else:
            print("All weights match perfectly!")

        # Evaluate the model
        model.eval()
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            log_every_n_steps=None  # This disables per-step logging
        )

        print("Testing on val loader")
        trainer.test(model, val_loader)
        print("Testing on test loader")
        trainer.test(model, test_loader)


if __name__ == "__main__":

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

    # Load configuration parameters
    args = OmegaConf.load('config_baseline.yaml')
    if deactivate_wandb:
        args.Training.wandb = False

    # Set random seed for reproducibility
    seed = args.Training.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    main(args)

    if args.Training.wandb:
        wandb.finish()
