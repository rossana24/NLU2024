# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py

import torch
import torch.nn as nn
import torch.optim as optim

from functions import train_loop, eval_loop, generate_sentence

import random
import math
import numpy as np
from omegaconf import OmegaConf

from utils import get_data_loaders, Lang
from model import LM_LSTM_proposed_model
from tqdm import tqdm
import copy
from datetime import datetime
import os
import json

try:
    import wandb

    deactivate_wandb = 0
except ImportError:
    print("WANDB is not available please install")
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

    if args.Training.eval_path == '':

        # Timestamp for naming the run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_run_name = f'{timestamp}_{args.Training.run_name}_{args.Training.tag}'

        # Create directory for checkpoints
        checkpoints_dir = os.path.join('.', 'bin', new_run_name)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Save the configuration parameters
        args_file_path = 'paramters.json'
        args_dict = OmegaConf.to_container(args, resolve=True)
        args_file_path = os.path.join(checkpoints_dir, args_file_path)
        with open(args_file_path, 'w') as args_file:
            json.dump(args_dict, args_file, indent=4)

        # Get data loaders for training, development, and testing
        train_loader, dev_loader, test_loader, lang = get_data_loaders(args)
        vocab_len = len(lang.word2id)

        # Initialize the model
        model = LM_LSTM_proposed_model(emb_size=args.Model.emb_size,
                                       hidden_size=args.Model.hid_size,
                                       output_size=vocab_len,
                                       device=device,
                                       weight_drop_locked_i=args.Model.dropout_locked_i,
                                       weight_drop_locked_h=args.Model.dropout_locked_h,
                                       weight_drop_locked_o=args.Model.dropout_locked_o,
                                       pad_index=lang.word2id["<pad>"],
                                       n_layers=args.Model.n_layers,
                                       tie_weights=args.Model.tie_weights,
                                       tbptt=args.Model.tbptt,
                                       tbptt_config=args.Model.tbptt_config,
                                       ).to(device)
        model.flatten_parameters()

        # Initialize wandb if enabled
        if args.Training.wandb:
            run_id = wandb.util.generate_id()
            print(f"Starting a new wandb run with id {run_id}")
            wandb.init(
                # set the wandb project where this run will be logged
                project="NLU_Project_LM_2",
                config=args_dict,
                tags=["LM", "Solution", args.Training.tag],
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

        # Set up learning rate scheduler to adjust the learning rate during training
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True)

        # Define loss functions for training and evaluation
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        # Initialize training parameters
        n_epochs = args.Training.num_epochs
        patience = args.Training.patience
        clip = args.Training.clip
        losses_train = []
        losses_dev = []
        pev_dev_list = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None

        # Progress bar for epochs
        pbar = tqdm(range(1, n_epochs))
        for epoch in pbar:
            # Train the model for one epoch
            loss = train_loop(data=train_loader,
                              optimizer=optimizer, model=model, criterion=criterion_train, device=device,
                              batch_size=args.Dataset.batch_size_train, clip=clip)

            # Evaluate the model and track losses and perplexity
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)

                if args.Model.tbptt:
                    losses_train.append(np.asarray(loss.cpu()).mean())
                else:
                    losses_train.append(np.asarray(loss).mean())

                # Handle optimizer state and evaluate the model
                if 't0' in optimizer.param_groups[0]:
                    # Handle ASGD optimizer

                    tmp = {}
                    for prm in model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer.state[prm]['ax'].clone()

                    ppl_dev, loss_dev = eval_loop(data=dev_loader,
                                                  eval_criterion=criterion_eval,
                                                  model=model,
                                                  device=device,
                                                  batch_size=args.Dataset.batch_size_valid)
                    pev_dev_list.append(ppl_dev)
                    losses_dev.append(np.asarray(loss_dev).mean())
                    pbar.set_description("PPL: %f" % ppl_dev)
                    print("\nPpa is {} and loss is {}".format(ppl_dev, np.asarray(loss_dev).mean()))

                    if args.Training.wandb:
                        wandb.log({"train/loss": loss,
                                   "validation/ppl": ppl_dev,
                                   "validation/loss": loss_dev,
                                   "train/lr": scheduler.get_last_lr()[0]},
                                  step=epoch)

                    # Save the best model based on validation perplexity
                    if ppl_dev < best_ppl:
                        best_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = args.Training.patience
                    else:
                        patience -= 1
                        print(f"Epoch did not improve, Patience is going down by 1 step to {patience}")

                    # Early stopping if patience runs out
                    if patience <= 0:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, "e_{}_best_model.pth".format(epoch)))
                        print("The patience has been achieve. Hard luck next time my friend. ")
                        break

                    for prm in model.parameters():
                        prm.data = tmp[prm].clone()

                else:
                    # Handle other optimizers (not ASGD)
                    ppl_dev, loss_dev = eval_loop(data=dev_loader,
                                                  eval_criterion=criterion_eval,
                                                  model=model,
                                                  device=device,
                                                  batch_size=args.Dataset.batch_size_valid)
                    losses_dev.append(np.asarray(loss_dev).mean())

                    if args.Training.wandb:
                        wandb.log({"train/loss": loss,
                                   "validation/ppl": ppl_dev,
                                   "validation/loss": loss_dev,
                                   "train/lr": scheduler.get_last_lr()[0]},
                                  step=epoch)
                    print("\nPpa is {} and loss is {}".format(ppl_dev, np.asarray(loss_dev).mean()))

                    if ppl_dev < best_ppl:
                        best_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = args.Training.patience
                    else:
                        patience -= 1
                        print(f"Epoch did not improve, Patience is going down by 1 step to {patience}")
                    if patience <= 0:  # Early stopping with patience
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, "e_{}_best_model.pth".format(epoch)))
                        print("The patience has been achieve. Hard luck next time my friend. ")
                        break

                    # Check if the current optimizer is SGD and the 't0' parameter is not present
                    # in the optimizer's parameters, indicating that ASGD is not currently used.

                    # ______ Optional: switch optimizer to ASGD if conditions are met
                    # Comment the lines below to disable switching to ASGD
                    if (args.Training.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and
                            (len(pev_dev_list) > args.Model.non_monotone_interval and
                             ppl_dev > min(pev_dev_list[:-args.Model.non_monotone_interval]))):
                        print('Switching to ASGD')

                        # Initialize a new ASGD optimizer with the same parameters as the previous SGD optimizer
                        optimizer = torch.optim.ASGD(model.parameters(), lr=args.Training.lr, t0=0, lambd=0.,
                                                     weight_decay=args.Training.weight_decay)

                    pev_dev_list.append(ppl_dev)
                    # ---------

            # Update learning rate scheduler after 10% of epochs
            if epoch > int(args.Training.num_epochs * 0.1):
                scheduler.step(loss_dev)

        # Save the best model after all epochs or early stopping
        torch.save(best_model.state_dict(), os.path.join(checkpoints_dir, "e_{}_best_model.pth".format(epoch)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'word2id': lang.word2id,
            # You can add more items here as needed
        }, os.path.join(checkpoints_dir, "best_checkpoint.pth"))

        # Load the best model and evaluate on the test set
        model = best_model.to(device)

        ppl_test, loss_test = eval_loop(data=test_loader,
                                        eval_criterion=criterion_eval,
                                        model=model,
                                        device=device,
                                        batch_size=args.Dataset.batch_size_test)
        print("Ppa is {} and loss is {}".format(ppl_test, np.asarray(loss_test).mean()))

        # Log test metrics to wandb if enabled
        if args.Training.wandb:
            wandb.log({"test/ppl": ppl_test,
                       "test/loss": loss_test}, step=epoch)

    else:
        checkpoint_path = os.path.join(args.Training.eval_path, "best_checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        load_lang = Lang.load(checkpoint)
        train_loader, dev_loader, test_loader, lang = get_data_loaders(args, load_lang)
        vocab_len = len(lang.word2id)

        # Define loss functions for training and evaluation
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        # Initialize the model
        best_model = LM_LSTM_proposed_model(emb_size=args.Model.emb_size,
                                            hidden_size=args.Model.hid_size,
                                            output_size=vocab_len,
                                            device=device,
                                            weight_drop_locked_i=args.Model.dropout_locked_i,
                                            weight_drop_locked_h=args.Model.dropout_locked_h,
                                            weight_drop_locked_o=args.Model.dropout_locked_o,
                                            pad_index=lang.word2id["<pad>"],
                                            n_layers=args.Model.n_layers,
                                            tie_weights=args.Model.tie_weights,
                                            tbptt=args.Model.tbptt,
                                            tbptt_config=args.Model.tbptt_config,
                                            ).to(device)
        best_model.flatten_parameters()
        best_model.load_state_dict(checkpoint['model_state_dict'])

        # Doing evaluation step on validation set
        ppl_test, loss_test = eval_loop(data=dev_loader,
                                        eval_criterion=criterion_eval,
                                        model=best_model,
                                        # weight of the best model? Ore just weigts of the latest epoch?
                                        device=device,
                                        batch_size=args.Dataset.batch_size_valid)
        print(ppl_test)

        # Doing evaluation step on test set
        ppl_test, loss_test = eval_loop(data=test_loader,
                                        eval_criterion=criterion_eval,
                                        model=best_model,
                                        # weight of the best model? Ore just weigts of the latest epoch?
                                        device=device,
                                        batch_size=args.Dataset.batch_size_test)
        print(ppl_test)

        # Example sentence generation
        print("\nGenerating sentences:")
        for start_sentence in ["the", "los angeles", "in the first half"]:
            generated_sentence = generate_sentence(best_model, start_sentence, lang, max_len=10, topk=5, unk=True, device=device)
            print(f"Start: '{start_sentence}' -> Generated: {' '.join(generated_sentence)}")


if __name__ == "__main__":
    # Load configuration from YAML file
    args = OmegaConf.load('config_model.yaml')

    if deactivate_wandb:
        # Disable WandB if the library is not available
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
