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
