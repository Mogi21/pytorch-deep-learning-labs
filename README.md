# PyTorch Deep Learning Fundamentals

This repository contains my coursework and lab submissions for Stockholm University (DSV), built using pure PyTorch. It demonstrates hands-on implementation of neural network architectures, custom data pipelines, optimization algorithms, regularization techniques, and advanced training protocols.

---

## 📂 Repository Structure & Contents

### Lab 1: PyTorch Basics & Neural Network Fundamentals
* **`01_pytorch_classification.ipynb`** (Multi-class Classification)
  * **Dataset:** Fashion MNIST (70,000 grayscale images, 10 classes)[cite: 2].
  * **Architecture:** Single-layer linear classifier mapping 784 input pixels directly to 10 class logits.
  * **Key Concepts:** Manual versus automatic differentiation (Autograd), cross-entropy loss derivation, and data pipeline optimization using custom PyTorch `Dataset` and `DataLoader` classes.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/01_pytorch_classification.ipynb)

* **`02_pytorch_regression.ipynb`** (Continuous Regression)
  * **Dataset:** California Housing Dataset.
  * **Architecture:** Linear regression network adapted for continuous target prediction.
  * **Key Concepts:** Feature standardization using scikit-learn, Mean Squared Error (`MSELoss`) optimization, and inference pipeline structuring for evaluation.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/02_pytorch_regression.ipynb)

---

### Lab 2: Training Optimization & Network Tuning
* **`03_pytorch_wine_quality.ipynb`** (Tabular Classification & Regularization)
  * **Dataset:** Wine Quality Dataset (Red and white Portuguese Vinho Verde wine physicochemical properties)[cite: 6].
  * **Architecture:** Multi-layer Sequential network (`nn.Sequential`) featuring linear transformation layers, ReLU activations, and Dropout layers for regularization.
  * **Key Concepts:** 
    * Investigation of optimization dynamics across loss functions (BCE vs. MSE) and L2 (Ridge) regularization parameters.
    * Advanced optimization strategy utilizing **AdamW** with weight decay.
    * Implementation of **Early Stopping** with validation loss tracking and automatic weight restoration to prevent overfitting.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/03_pytorch_wine_quality.ipynb)

---

## 🛠️ Tech Stack
* **Framework:** PyTorch (`torch`, `torch.nn`, `torch.utils.data`)
* **Data Processing & Metrics:** NumPy, Pandas, Scikit-Learn (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `f1_score`)
* **Visualization:** Matplotlib, Seaborn, Graphviz
