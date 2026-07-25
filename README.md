# AI Studio

An end-to-end Machine Learning and Deep Learning platform built with **Python**, **Scikit-learn**, and **PyTorch**.

AI Studio allows users to upload their own datasets, select the machine learning task, choose from recommended algorithms, train models, evaluate performance, visualize results, and save trained models—all through a single application.

---

## Features

### Dataset Support

- Image Datasets
- CSV Datasets
- Automatic Dataset Validation
- Automatic Dataset Information

---

### Supported Tasks

#### Image Datasets

- Image Classification
- Object Detection *(Coming Soon)*
- Image Segmentation *(Coming Soon)*

#### CSV Datasets

- Classification
- Regression
- Clustering
- Time Series Forecasting

---

## Model Support

### Classical Machine Learning (Scikit-learn)

#### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Naive Bayes

#### Regression

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

#### Clustering

- K-Means
- Hierarchical Clustering
- DBSCAN

---

### Deep Learning (PyTorch)

#### Image Classification

- Custom CNN
- ResNet
- EfficientNet

#### Time Series Forecasting

- RNN
- LSTM
- GRU
- Transformer *(Coming Soon)*

#### Tabular Deep Learning

- Feedforward Neural Network (MLP)

---

## Workflow

```
Upload Dataset
       │
       ▼
Detect Dataset Type
(Image / CSV)
       │
       ▼
Validate Dataset
       │
       ▼
Choose Task
       │
       ├── Classification
       ├── Regression
       ├── Clustering
       ├── Time Series Forecasting
       └── Image Classification
               │
               ▼
Recommend Models
       │
       ├── Classical Machine Learning
       └── Deep Learning
               │
               ▼
Configure Training
               │
               ▼
Train Model
               │
               ▼
Evaluate Performance
               │
               ▼
Visualize Metrics
               │
               ▼
Save Best Model
               │
               ▼
Inference
```

---

## Technology Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Deep Learning

- PyTorch
- TorchVision

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### User Interface

- Streamlit

---

## Project Structure

```
AI-Studio/

│── app/
│   ├── dataset/
│   ├── models/
│   ├── trainer/
│   ├── metrics/
│   ├── visualization/
│   ├── inference/
│   ├── ui/
│   └── config/
│
│── datasets/
│── saved_models/
│── experiments/
│── logs/
│
│── requirements.txt
│── main.py
│── README.md
```

---

## Roadmap

### Version 1.0

- Dataset Validation
- Image Classification
- Classification
- Regression
- Clustering
- Time Series Forecasting
- Model Evaluation
- Model Saving
- Inference

### Version 2.0

- Object Detection
- Image Segmentation
- Transfer Learning
- Hyperparameter Tuning
- Explainable AI (SHAP)
- Experiment Tracking

### Version 3.0

- Transformers
- Vision Transformers
- Large Language Models
- RAG
- AutoML
- Distributed Training

---

## Status

- [x] Project Planning
- [x] Architecture Design
- [ ] Dataset Module
- [ ] Model Factory
- [ ] Trainer
- [ ] Metrics
- [ ] Visualization
- [ ] Inference

---

## License

MIT License