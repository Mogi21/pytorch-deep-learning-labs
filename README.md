# PyTorch Deep Learning Fundamentals

This repository contains my coursework and lab submissions for Stockholm University (DSV), built using pure PyTorch. It demonstrates hands-on implementation of neural network architectures, custom data pipelines, optimization algorithms, regularization techniques, and advanced training protocols.

---

## 📂 Repository Structure & Contents

### Lab 1: PyTorch Basics & Neural Network Fundamentals
* **`01_pytorch_classification.ipynb`** (Multi-class Classification)
  * **Dataset:** Fashion MNIST (70,000 grayscale images, 10 classes).
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
  * **Dataset:** Wine Quality Dataset (Red and white Portuguese Vinho Verde wine physicochemical properties).
  * **Architecture:** Multi-layer Sequential network (`nn.Sequential`) featuring linear transformation layers, ReLU activations, and Dropout layers for regularization.
  * **Key Concepts:** 
    * Investigation of optimization dynamics across loss functions (BCE vs. MSE) and L2 (Ridge) regularization parameters.
    * Advanced optimization strategy utilizing **AdamW** with weight decay.
    * Implementation of **Early Stopping** with validation loss tracking and automatic weight restoration to prevent overfitting.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/03_pytorch_wine_quality.ipynb)

---

### Lab 3: Convolutional Neural Networks & Computer Vision
* **`04_pytorch_cnns_cifar10.ipynb`** (Image Classification & Transfer Learning)
  * **Dataset:** CIFAR-10 (60,000 32x32 RGB images across 10 classes).
  * **Architectures:** 
    * Custom CNN with sequential convolutional/pooling blocks, Batch Normalization, and Dropout.
    * **ResNet18** (Transfer learning via pre-trained weights from ImageNet).
  * **Key Concepts:** 
    * Design and implementation of 2D convolutional layers, max-pooling operations, and programmatic spatial shape tracking algorithms.
    * Real-time image **Data Augmentation** (`RandomCrop`, `RandomHorizontalFlip`) to improve model generalization and prevent overfitting.
    * **Transfer Learning**: Freezing deep feature extraction backbones and fine-tuning isolated classification heads for rapid convergence.
    * Engineering a custom PyTorch `Dataset` subclass to ingest and normalize raw NumPy arrays for inference.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/04_pytorch_cnns_cifar10.ipynb)

---

### Lab 4: Recurrent Neural Networks & Time-Series Analysis
* **`05_pytorch_rnns_timeseries.ipynb`** (Forecasting & Sequence Classification)
  * **Datasets:** 
    * Synthetic superimposed chaotic continuous time-series (Forecasting).
    * Z-scored dynamical systems dataset featuring 100-step sequences mapped to 4 distinct classes (Classification).
  * **Architectures:** 
    * `ScratchRNNCell`: Manual implementation of a recurrent cell explicitly defining temporal hidden state propagation.
    * Gated Sequence Models: Leveraging PyTorch's optimized `nn.LSTM` and `nn.GRU` to combat the vanishing gradient problem inherent in BPTT (Backpropagation Through Time).
    * Deep Sequence Classifier: Multi-layer RNN integrating temporal output slicing, Batch Normalization (`nn.BatchNorm1d`), and Dropout logic.
  * **Key Concepts:** 
    * Analyzing BPTT gradient degradation over long sequences (e.g., $t=200$).
    * Utilizing Gradient Clipping (`clip_grad_norm_`) to stabilize explosive recurrences.
    * Employing strict walk-forward temporal validation splits to strictly avoid future-data leakage during standardization and evaluation.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/05_pytorch_rnns_timeseries.ipynb)

---

### Lab 5: Transformers & Natural Language Processing
* **`06_pytorch_transformers.ipynb`** (Embeddings, Fine-Tuning & Autoregression)
  * **Datasets:** AG News Corpus (sub-sampled), Custom Text Classification datasets.
  * **Architectures:** 
    * Pre-trained Base `BERT` (`bert-base-uncased`) for feature extraction and Masked Language Modeling (MLM).
    * `RoBERTa` (`roberta-base`) configured with a Sequence Classification head.
  * **Key Concepts:** 
    * Accessing and interpreting contextualized outputs (pooler outputs, hidden states, attention matrices) from the Hugging Face `transformers` ecosystem.
    * Engineering text similarity pipelines leveraging **Cosine Similarity** on `[CLS]` token pooler embeddings.
    * **Fine-Tuning Strategies**: Freezing backbone parameters where necessary and applying `AdamW` optimizers with conservative learning rates (e.g., `2e-5`) to specialize pre-trained transformers on downstream classification tasks.
    * Simulating **Causal Autoregressive Text Generation** by iteratively masking and predicting sequences using an MLM head.
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mogi21/pytorch-deep-learning-labs/blob/main/06_pytorch_transformers.ipynb)

---

## 🛠️ Tech Stack
* **Framework:** PyTorch (`torch`, `torch.nn`, `torch.utils.data`), Hugging Face Transformers (`AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, `AutoModelForMaskedLM`)
* **Data Processing & Metrics:** NumPy, Pandas, Scikit-Learn (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `f1_score`, `cosine_similarity`), Hugging Face Datasets
* **Visualization:** Matplotlib, Seaborn, Graphviz
