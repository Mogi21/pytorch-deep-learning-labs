# PyTorch Deep Learning Fundamentals

This repository contains PyTorch implementations of neural network architectures for both classification and regression tasks. It demonstrates a foundational understanding of computational graphs, custom data pipelines, and the mathematical adaptations required to shift between predicting probabilities and continuous values.

## 📁 Repository Contents

### 1. Multi-Class Classification: Fashion MNIST (`01_pytorch_classification.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/01_pytorch_classification.ipynb)
*   **Custom Data Pipeline:** Implements the `torch.utils.data.Dataset` and `DataLoader` APIs to dynamically load images from disk, preventing RAM overflow during training.
*   **Architecture:** A single-layer linear network ($\text{Linear} \rightarrow \text{Softmax} \rightarrow \text{Cross-Entropy}$) designed to baseline multi-class classification on the Fashion MNIST dataset.
*   **Mathematical Verification:** Includes a manual NumPy implementation of the forward pass, softmax, and backpropagation (chain rule) to verify PyTorch's `autograd` engine to machine precision.

### 2. Continuous Regression: California Housing (`02_pytorch_regression.ipynb`)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/02_pytorch_regression.ipynb)
*   **Architectural Shift:** Adapts the classification network into a regression model by replacing the 10-class output layer with a single continuous output node.
*   **Loss Function Adaptation:** Transitions from Cross-Entropy (probability distributions) to Mean Squared Error (MSE) for continuous value prediction.
*   **Data Scaling:** Utilizes `StandardScaler` to normalize feature distributions, preventing vanishing/exploding gradients during the backpropagation phase. 
*   *Performance:* Successfully tuned to achieve an MSE of < 0.5 on the held-out test set.
