# LM part_2: Project Guide

To run the model, execute the main script.

### Configuration
`config_bert.yaml` configuration file manages:
- Model specifications: Set model hyperparameters.
- Dataset definition: Define paths to training and test datasets.
- Tbptt mode: Specify the use of the TBPTT technique.

`model.py` file contains the code for defining the model architecture. It includes:
- Model Definition: Specifies how the model is structured and its components.
- Custom optimization and regularization techniques: the implementation follows the one of the model that achieved the best performance. However, all techniques are implemented in a flexible structure, allowing for easy modification or addition as needed.

`utils.py` file contains utility functions for:
- Data Loading: Functions to load and preprocess data.
- Preprocessing: Includes handling the OOV tokens and  the challenge of dealing with variable-length sequences in text data.

`functions.py` file containing the code for training and evaluating the model and the implementation of the additional evaluation metric.