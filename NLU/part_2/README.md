# NLU part_2: Project Guide

To run the model, execute the main script.

### Configuration
`config_bert.yaml` configuration file manages:

- Model specifications: Set model hyperparameters.
- Dataset definition: Define paths to training and test datasets.
- Evaluation mode: Specify the path to a saved model if you want to evaluate an existing model.


`model.py` file contains the code for defining the model architecture. It includes:

- Model Definition: Specifies how the model is structured and its components.
- Custom Detokenization: Implements a custom_detokenize function to address sub-tokenization issues based on a contraction map.
- Training and Evaluation loops.

`utils.py` file contains utility functions for:
- Data Loading: Functions to load and preprocess data.
- Preprocessing: Includes handling of text data and vocabulary creation.
- Validation Set Creation: During preprocessing, a validation set is created from 10% of the training data. This split helps in evaluating the model's performance during training.
- Vocabulary Management: A vocabulary variable is created to store the vocabulary from the training set. For OOV data, the token is set as UNK in the validation and test sets.

