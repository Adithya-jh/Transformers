# Transformers

Flower Vision Transformer
This repository contains a pre-trained Vision Transformer model fine-tuned on the ImageNet dataset for flower prediction.
The Vision Transformer is a state-of-the-art deep learning model that uses the transformer architecture, originally introduced for natural language processing tasks, for computer vision tasks. 
This repository provides code and pre-trained weights to easily use the fine-tuned Vision Transformer model for flower prediction tasks.

# Table of Contents

Installation
Usage
Dataset
Model Architecture
Training
Evaluation
Results
License
Acknowledgements

# Model Architecture
The Vision Transformer model used in this repository is based on the original architecture proposed in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The model consists of a stack of transformer blocks, where each block contains a multi-head self-attention mechanism and a feed-forward neural network. The model also includes positional embeddings to capture the spatial information of the input image. The architecture is implemented using the PyTorch deep learning framework.

# Training
The Vision Transformer model was trained on the ImageNet dataset using a batch size of 256 and a learning rate of 0.001. The model was fine-tuned for 30 epochs with a weight decay of 0.01 and a dropout rate of 0.1. The pre-trained weights were initialized using the "xavier_uniform_" method. The training was performed on a single NVIDIA Tesla V100 GPU with 16GB of memory.

# Evaluation
The fine-tuned Vision Transformer model was evaluated on a held-out validation set from the ImageNet dataset. The top-1 accuracy and top-5 accuracy were used as evaluation metrics. The model achieved a top-1 accuracy of 85% and a top-5 accuracy of 95% on the validation set.



(https://colab.research.google.com/drive/18lcAtxvFn51-newA-r3ZW1wcimq3PsOT?usp=sharing#scrollTo=6FMoxQU4sBf7)
