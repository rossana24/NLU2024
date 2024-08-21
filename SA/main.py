import os
import random
import json
from omegaconf import OmegaConf
import torch
import numpy as np
from datetime import datetime
from transformers import BertConfig

from model import BertForSpanAspectExtraction
from utils import get_dataloaders
from functions import run_train_epoch, prepare_optimizer, run_valid_epoch, save_checkpoint


def main(config):
    """
    Summary:
    Main function to set up and run the training and evaluation of the BERT model for span-based aspect extraction.

    Input:
     * config (OmegaConf object): Configuration parameters for the model, training, and data
    """
    # Determine the device to run the model on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamp for naming the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'{timestamp}_{config.Training.run_name}_{config.Training.tag}'

    # Create directory for checkpoints
    checkpoints_dir = os.path.join('.', 'bin', new_run_name)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Save the configuration parameters
    args_file_path = 'paramters.json'
    args_dict = OmegaConf.to_container(config, resolve=True)
    args_file_path = os.path.join(checkpoints_dir, args_file_path)
    with open(args_file_path, 'w') as args_file:
        json.dump(args_dict, args_file, indent=4)

    # Get data loaders for training and validation
    train_examples, train_dataloader = get_dataloaders(config, train=True)
    val_examples, val_dataloader = get_dataloaders(config, train=False)

    global_step = 0
    # Load BERT configuration with the specified task name
    config_bert = BertConfig.from_pretrained('bert-base-uncased', finetuning_task=config.Dataset.data_set_name)

    # Get the maximum sequence length for BERT
    max_sequence_length = config_bert.max_position_embeddings
    print(f"The maximum sequence length for BERT is {max_sequence_length} tokens.")

    # Initialize the model with pre-trained BERT weights
    model = BertForSpanAspectExtraction.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
                                                        config=config_bert).to(device)

    # Prepare the optimizer
    optimizer = prepare_optimizer(config, model)

    model.train()
    save_checkpoints_steps = 1  # Save checkpoints every epoch
    for epoch in range(int(config.Training.num_epochs)):
        print("***** Epoch: {} *****".format(epoch + 1))

        # Run one training epoch
        run_train_epoch(global_step, model, train_dataloader, optimizer, device)

        # Validate the model after each epoch
        run_valid_epoch(model, val_dataloader, device, threshold=config.Dataset.threshold)#, unique_id_to_feature) validation

        # Save the model checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoints_dir, f'model_epoch_{epoch + 1}.pth')
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

if __name__ == "__main__":
    """
    Summary:
    Entry point of the script. Loads the configuration, sets up the environment,
    and runs the main function to start training and evaluation.
    """
    # Set environment variable to help with CUDA error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Load configuration from YAML file
    args = OmegaConf.load('config_bert.yaml')

    # Set random seeds for reproducibility
    seed = args.Training.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    # Run the main function with loaded arguments
    main(args)
