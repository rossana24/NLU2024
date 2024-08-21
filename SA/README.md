# Assignment Overview: SA

The goal of the project is to implement a model based on BERT pre-trained language model for the Aspect-Based Sentiment Analysis (ABSA) task, specifically focused on extracting aspect terms.

---
## Details
- **Laptop partition of SemEval2014 task 4 dataset** 
- **Evaluation Metrics :**
  - **F1 score** 
  - **Precision**
  - **Recall** 
---

## Project Guide

To run the model, execute the main script.

### Configuration
`config_bert.yaml` configuration file manages:

- Model specifications: Set model hyperparameters.
- Dataset definition: Define paths to training and test datasets.

`utils.py` file contains utility functions for:
- Data Loading: Functions to load and preprocess data.
- Preprocessing: Includes handling of text data and tokenization.
- Aspect terms Management: Functions to detect and extract start and end indices of each sentiment span.


`model.py` file contains the code for defining the model architecture and the distant cross-entropy loss.

`functions.py` file contains functions for training and evaluation loops and conversion to BIO and BIOES tags.

`evals.py` file contains code for evaluation based on BIOES notation.

