<div align="center">

# HybridNER: A Transformer and LSTM-based Named Entity Recognition Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
<br> **HybridNER** is a Named Entity Recognition (NER) model that combines the strengths of Transformer and LSTM architectures. This hybrid approach allows the model to effectively capture both long-term dependencies and complex contextual relationships within text sequences, resulting in improved accuracy and performance in NER tasks.
</div>

## Model Overview & Features
The HybridNER model integrates Transformer layers for their exceptional capability in capturing context-aware relationships and LSTM layers for their strength in managing sequential data dependencies. This combination enables the model to achieve superior performance in identifying and classifying named entities within text.
-  **Built from Scratch:** Entirely implemented using PyTorch, ensuring a deep understanding of the underlying mechanics and complete control over the architecture.
-  **Hybrid Architecture:** Combines Transformer and LSTM layers to leverage the strengths of both sequential and contextual data processing.
-  **Easy Extensibility:** The modular design allows for straightforward integration of additional features and customization, making it highly adaptable for various NER tasks.

## Named Entity Recognition (NER) Task
Named Entity Recognition (NER) is a crucial task in Natural Language Processing (NLP) that involves identifying and classifying key information (entities) in text into predefined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. NER is used in a variety of applications, including information retrieval, question answering, and automated summarization.
<p align="center">
** Example of NER **
   <img src="https://github.com/Tro-fish/HybridNER-A-Transformer-and-LSTM-based-Named-Entity-Recognition-Model/assets/79634774/1e122bbf-e7cb-4d48-9871-0618376ba14a" alt="Description of the image" width="100%" />
</p>
Consider the sentence: “Michael Jeffery Jordan was born in Brooklyn, New York.” In this sentence, the NER model should recognize and categorize “Michael Jeffery Jordan” as a person (PER), “Brooklyn” and “New York” as locations (LOC).

## Model Architecture
<p align="center">
  <img src="https://github.com/Tro-fish/HybridNER-A-Transformer-and-LSTM-based-Named-Entity-Recognition-Model/assets/79634774/1d6d8dbe-577c-46e7-a15c-406df883994b" alt="Description of the image" width="100%" />
</p>

## Repository Structure
- NER_models.py: Definition of the NER model architecture, combining Transformer and LSTM layers.
- NER_dataset.py: Script to handle the dataset loading and preprocessing for Named Entity Recognition (NER).
- train.py: Script to train the NER model.
- inference.py: Script to perform inference using the trained NER model.
- utils.py: Utility functions and configurations for model training and evaluation.

## Model Configuration & Test Results

<table>
  <tr>
    <td>

### Model Configuration

| Parameter            | Value     |
|----------------------|-----------|
| seed                 | 12        |
| batch_size           | 64        |
| lr                   | 3e-4      |
| weight_decay         | 0.01      |
| hidden_size          | 256       |
| num_heads            | 4         |
| num_encoder_layers   | 6         |
| hidden_dropout_prob  | 0.1       |
| use_lstm             | True      |
| num_epochs           | 50        |
| vocab_size           | 30522     |
| pad_token_id         | 0         |
| num_labels           | 9         |

</td>
    <td>

### Test Results

| Metric     | Score       |
|------------|-------------|
| Precision  | 0.793670    |
| Recall     | 0.763884    |
| F1 Score   | 0.778492    |
| Accuracy   | 0.952845    |

</td>
  </tr>
</table>

## Model Configuration & Test Results

| Parameter            | Value     |       | Metric     | Score       |
|----------------------|-----------|       |------------|-------------|     
| batch_size           | 64        |       | Accuracy   | 0.952845    |
| lr                   | 3e-4      |       | F1 Score   | 0.778492    |
| weight_decay         | 0.01      |       | Recall     | 0.763884    |
| hidden_size          | 256       |       | Precision  | 0.793670    |
| num_heads            | 4         |
| num_encoder_layers   | 6         |
| hidden_dropout_prob  | 0.1       |
| num_epochs           | 30        |
| num_labels           | 9         |

## Test Results

| Metric     | Score       |
|------------|-------------|
| Precision  | 0.793670    |
| Recall     | 0.763884    |
| F1 Score   | 0.778492    |
| Accuracy   | 0.952845    |
