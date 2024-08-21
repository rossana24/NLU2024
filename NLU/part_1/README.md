# NLU part_1: Project Guide

To run the model, execute the main script.

### Configuration
`config_baseline.yaml` configuration file manages:

- Model specifications: Set model hyperparameters.
- Dataset definition: Define paths to training and test datasets.
- Evaluation mode: Specify the path to a saved model if you want to evaluate an existing model.

`utils.py` file contains utility functions for:
- Data Loading: Functions to load and preprocess data.
- Preprocessing: Includes handling of text data and vocabulary creation.
- Validation Set Creation: During preprocessing, a validation set is created from 10% of the training data. This split helps in evaluating the model's performance during training.

`model.py` file contains the code for defining the model architecture, training and evaluation steps. 

`functions.py` file contains the code to perform dataset analysis.
