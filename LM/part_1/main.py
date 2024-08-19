# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py

import torch.nn as nn
import torch.optim as optim

from functions import train_loop, eval_loop, init_weights, generate_sentence

import random
import math
from omegaconf import OmegaConf
from utils import *
from model import LM_RNN, LM_LSTM
from tqdm import tqdm
import copy
from datetime import datetime
import os
import json


# Attempt to import wandb for experiment tracking
try:
    import wandb
    deactivate_wandb = 0
except ImportError:
    print("WANDB is not available, please install")
    print("use command: pip3 install wandb")
    deactivate_wandb = 1


def main(args):
    """
    Summary:
    Main function to set up and run the training and evaluation of the language model.

    Input:
     * args (OmegaConf object): Configuration parameters for the model, training, and data
    """

    # Determine the device to run the model on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This code is running on: {device}")

    # Timestamp for naming the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'{timestamp}_{args.Training.run_name}_{args.Training.tag}'

    # Create directory for checkpoints
    checkpoints_dir = os.path.join('.', 'bin', new_run_name)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Save the configuration parameters
    args_file_path = 'parameters.json'
    args_dict = OmegaConf.to_container(args, resolve=True)
    args_file_path = os.path.join(checkpoints_dir, args_file_path)
    with open(args_file_path, 'w') as args_file:
        json.dump(args_dict, args_file, indent=4)

    # Get data loaders for training, development, and testing
    train_loader, dev_loader, test_loader, lang = get_data_loaders(args)

    # ----- DATASET PREPROCESSING ------
    # Tokenized data
    #tokenized_train = [sent.split() for sent in read_file(args.Dataset.train_dataset_path)]
    #tokenized_valid = [sent.split() for sent in read_file(args.Dataset.valid_dataset_path)]
    #tokenized_test = [sent.split() for sent in read_file(args.Dataset.test_dataset_path)]

    #calculate_statistics([tokenized_train, tokenized_valid, tokenized_test], ['Train', 'Valid', 'Test'])
    #calculate_top_word_frequencies(tokenized_train, tokenized_valid, tokenized_test, top_n=10)

    vocab_len = len(lang.word2id)
    # ________

    # Initialize the model based on the specified type
    if args.Model.model_type == 'baseline_LSTM':
        model = (LM_LSTM(emb_size=args.Model.emb_size, hidden_size=args.Model.hid_size, output_size=vocab_len,
                         out_dropout=args.Model.dropout_lstm, emb_dropout=args.Model.dropout_emb,
                         pad_index=lang.word2id["<pad>"], n_layers=args.Model.n_layers).
                 to(device))
    else:
        model = LM_RNN(emb_size=args.Model.emb_size, hidden_size=args.Model.hid_size, output_size=vocab_len,
                       pad_index=lang.word2id["<pad>"], n_layers=args.Model.n_layers).to(device)
        model.apply(init_weights)

    # Initialize wandb if enabled
    if args.Training.wandb:
        run_id = wandb.util.generate_id()
        print(f"Starting a new wandb run with id {run_id}")
        wandb.init(
            project="NLU_Project_LM_1",
            config=args_dict,
            tags=["LM_1", args.Training.tag],
        )
        if args.Training.watch_wandb:
            wandb.watch(model, criterion='gradients', log_freq=100)
        wandb.run.name = new_run_name

    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')

    # Choose optimizer based on configuration
    if args.Training.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.Training.lr, weight_decay=args.Training.weight_decay)
    elif args.Training.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.Training.lr, weight_decay=args.Training.weight_decay)
    elif args.Training.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.Training.lr, weight_decay=args.Training.weight_decay)
    else:
        assert False, f"Unsupported optimizer type: {args.Training.optimizer}"

    # Define the loss criteria for training and evaluation
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Initialize training parameters
    n_epochs = args.Training.num_epochs
    patience = args.Training.patience
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    # Progress bar for epochs
    pbar = tqdm(range(1, n_epochs))
    for epoch in pbar:
        # Train the model for one epoch
        loss = train_loop(args, train_loader, optimizer, criterion_train, model, device)

        # Evaluate the model and track losses and perplexity
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(args, dev_loader, criterion_eval, model, device)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            # Log metrics to wandb if enabled
            if args.Training.wandb:
                wandb.log({"train/loss": loss,
                           "validation/ppl": ppl_dev,
                           "validation/loss": loss_dev}, step=epoch)

            # Save the best model based on validation perplexity
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = args.Training.patience
            else:
                patience -= 1

            # Early stopping if patience runs out
            if patience <= 0:
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, "e_{}_best_model.pth".format(epoch)))
                print("The patience has been reached. Stopping training.")
                break

    # Save the best model after all epochs or early stopping
    torch.save(best_model.state_dict(), os.path.join(checkpoints_dir, "e_{}_best_model.pth".format(epoch)))

    # Load the best model and evaluate on the test set
    model = best_model.to(device)
    ppl_test, loss_test = eval_loop(args, test_loader, criterion_eval, model, device)
    print("PPL: {} and loss: {}".format(ppl_test, np.asarray(loss_test).mean()))

    # Log test metrics to wandb if enabled
    if args.Training.wandb:
        wandb.log({"test/ppl": ppl_test,
                   "test/loss": loss_test}, step=epoch)

    # Example sentence generation
    print("\nGenerating sentences:")
    for start_sentence in ["the", "los angeles", "in the first half"]:
        generated_sentence = generate_sentence(model, start_sentence, lang, max_len=10, topk=5, unk=True, device=device)
        print(f"Start: '{start_sentence}' -> Generated: {' '.join(generated_sentence)}")



if __name__ == "__main__":

    # Load configuration from YAML file
    args = OmegaConf.load('config_baseline_LSTM.yaml')

    if deactivate_wandb:
        # Disable WandB if the library is not availabley
        args.Training.wandb = False

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

    if args.Training.wandb:
        wandb.finish()
