# Assignment Overview: LM

The goal of the project is to implement language model based on LSTMs and to improve its performance through some regularization techniques to achieve a perplexity score less than 250.

### Details
- **PennTreeBank Dataset** 
- **Evaluation Metrics :**
  - **Perplexity** 
  - **Subjective metric to evaluate the models' prediction ability through generated sequences of words based on a seed sentence** 
---

### Part 1: Enhancing the Baseline Language Model

The goal is to modify the baseline language model, a LSTM network, 
incrementally by adding two dropout layers and using the AdamW optimizer. 

---

### Part 2: Fine-Tuning the Baseline model 

This part involves fine-tuning the Baseline language model investigating different 
optimization and regularization techniques like Weight Tying, Variational Dropout (no DropConnect), Non-monotonically Triggered AvSGD
 and Truncated BackPropagation Through Time.


---
