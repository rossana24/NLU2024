import os
import random
import json
from omegaconf import OmegaConf
import torch
import numpy as np
from datetime import datetime
from utils import get_data_loaders, Lang
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import NluModelBert

# Attempt to import wandb for experiment tracking
try:
    import wandb
    deactivate_wandb = 0
except ImportError:
    print("WANDB is not available please install")
    print("use command: pip3 install wandb")
    deactivate_wandb = 1

PAD_TOKEN = 0


def main(config):
    """
    Main function to handle training and evaluation of the model.

    Args:
      config (omegaconf.dictconfig.DictConfig): Configuration object containing parameters for training and evaluation.
    """

    if config.Training.eval_path == "":

        # Define the run name and create the directory for saving checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_run_name = f'{timestamp}_{config.Training.run_name}_{config.Training.tag}'
        checkpoints_dir = os.path.join('.', 'bin', new_run_name)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Save configuration parameters to a JSON file
        args_file_path = os.path.join(checkpoints_dir, 'parameters.json')
        args_dict = OmegaConf.to_container(config, resolve=True)
        with open(args_file_path, 'w') as args_file:
            json.dump(args_dict, args_file, indent=4)

        # Initialize wandb
        wandb_logger = WandbLogger(name=new_run_name, project=config.Training.project_name, config=args_dict,
                                   log_model=False)
        wandb_logger.experiment.config.update({'x_axis': 'epoch'})

        # Data preparation: Load data loaders, language model, and tokenizer
        train_loader, val_loader, test_loader, lang_model, bert_tokenizer = get_data_loaders(config)

        # Calculate total steps and warmup steps for the learning rate scheduler
        t_total = len(train_loader) // config.Training.gradient_accumulation_steps * config.Training.num_epochs
        warm_up_step = len(train_loader) * config.Training.warmup_steps

        # Print dataset and dataloader statistics
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
            filename='best_checkpoint',  # Custom filename without the epoch/step details
            monitor='val/f1',  # Monitor the validation F1 score
            mode='max',  # Save the model with the highest validation F1 score
            save_top_k=1,  # Only keep the best model
            save_last=False,  # Don't save the last epoch, only the best one
            save_weights_only=True,  # Save the entire model (including optimizer state, etc.)
        )
        early_stop_callback = EarlyStopping(
            monitor='val/f1',
            min_delta=0.00,
            patience=config.Training.patience,
            verbose=True,
            mode='max'
        )

        # Initialize the model
        model = NluModelBert(args=config, total_num_steps=t_total, warm_up_steps=warm_up_step,
                             lang_model=lang_model,
                             tokenizer=bert_tokenizer,
                             )

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Setup PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=config.Training.num_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            gradient_clip_val=config.Training.clip,
            devices=1 if torch.cuda.is_available() else None,
            logger=wandb_logger,
            callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
            log_every_n_steps=1)

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Test the model
        trainer.test(model, test_loader)

    else:
        # Load model and language model from checkpoint
        checkpoint_path = os.path.join(config.Training.eval_path, "best_checkpoint.ckpt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        load_lang = Lang.load(checkpoint)
        train_loader, val_loader, test_loader, lang_model, bert_tokenizer = get_data_loaders(config, load_lang)
        t_total = len(train_loader) // config.Training.gradient_accumulation_steps * config.Training.num_epochs
        warm_up_step = len(train_loader) * config.Training.warmup_steps
        load_args = checkpoint['args']

        # Initialize model from checkpoint
        model = NluModelBert.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=True,
                                                  args=load_args,
                                                  total_num_steps=t_total,
                                                  warm_up_steps=warm_up_step,
                                                  lang_model=load_lang,
                                                  tokenizer=bert_tokenizer)

        # Validate weights match between loaded checkpoint and model
        checkpoint_state_dict = model.state_dict()
        original_state_dict = checkpoint['state_dict']
        for key in original_state_dict:
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
            log_every_n_steps=None
        )

        print("Testing on val loader")
        trainer.test(model, val_loader)
        print("Testing on test loader")
        trainer.test(model, test_loader)


if __name__ == "__main__":

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Load configuration parameters
    args = OmegaConf.load('config_bert.yaml')
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

    # Execute the main function with loaded arguments
    main(args)

    if args.Training.wandb:
        wandb.finish()
