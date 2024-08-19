# LM part_1: Project Guide

To run the model, execute the main script.

### Configuration
`config_baseline.yaml` configuration file manages:

- Model specifications: Set model hyperparameters.
- Dataset definition: Define paths to training, valid and test datasets.

`utils.py` file contains utility functions for:
- Dataset analysis: Functions to analise the dataset statistics.
- Data Loading: Functions to load and preprocess data.
- Preprocessing: Includes handling the OOV tokens and  the challenge of dealing with variable-length sequences in text data.


`model.py` file contains the code for defining the model architecture. 

`functions.py` file containing the code for training and evaluating the model and the implementation of the additional evaluation metric.