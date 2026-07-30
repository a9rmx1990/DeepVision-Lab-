---
name: deepvisionlab-development
description: Use this skill whenever working on the DeepVisionLab project — an end-to-end AI platform for uploading datasets, training classical ML and deep learning models, evaluating results, visualizing metrics, saving models, and running inference. Trigger this any time the user asks to add a feature, fix a bug, write a module, refactor code, add a new model/task/metric, extend the dataset pipeline, or touch anything under the DeepVisionLab project structure (app/dataset, app/models, app/trainer, app/metrics, app/visualization, app/inference, app/ui, app/config, app/utils). Also trigger for questions about project architecture, coding standards, commit style, or where new code should live within this codebase. Always consult this skill before writing or editing any DeepVisionLab code, even for small changes, to keep the codebase consistent with the established architecture.
---

# DeepVisionLab Development

Guidance for building and extending DeepVisionLab, a modular, long-term AI platform. Use this skill to keep every change consistent with the project's architecture, pipelines, and coding standards — don't improvise structure that conflicts with what's below.

## Before writing code

1. Identify which pipeline stage the request belongs to (see Dataset Pipeline / Model Pipeline below) and place new code in the matching module — never mix stages together.
2. Check whether similar logic already exists elsewhere in the module (e.g. another preprocessing step, another metric) and reuse/extend it instead of duplicating.
3. Confirm the change respects the "Never" rules at the end of this file before finishing.

## Project Structure

```
DeepVisionLab/
  app/
    dataset/        # loader, validator, detector, splitter, preprocessing
    models/          # model factory + architectures (classical + deep learning)
    trainer/         # training loop, checkpoints, early stopping
    metrics/         # classification / regression / clustering metrics
    visualization/    # loss/accuracy curves, confusion matrix, ROC/PR curves
    inference/       # load model, predict on new data
    ui/              # Streamlit pages only — no business logic here
    config/          # settings, constants, paths
    utils/           # shared helpers
  datasets/          # uploaded/raw datasets
  saved_models/       # trained model artifacts
  experiments/        # experiment run outputs
  logs/               # training/inference logs
  main.py
  requirements.txt
  README.md
```

New files go in the module matching their responsibility. If a request doesn't clearly map to one of these modules, ask where it belongs rather than guessing a new top-level folder.

## Dataset Pipeline

Every uploaded dataset flows through this exact sequence — implement or extend it in this order, don't skip stages:

```
Upload Dataset → Loader → Validator → Detector → Splitter → Preprocessing → Dataset Object
```

| Stage | File | Responsibility |
|---|---|---|
| Load | `dataset/loader.py` | Load CSV datasets; load image datasets |
| Validate | `dataset/validator.py` | Validate dataset, check missing files, check corrupted images, check invalid CSV |
| Detect | `dataset/detector.py` | Detect dataset type: CSV or Image |
| Split | `dataset/splitter.py` | Create training / validation / test sets |
| Preprocess | `dataset/preprocessing.py` | Label encoding, one-hot encoding, scaling, image transformations |

Preprocessing logic must live only in `preprocessing.py` and be reused everywhere it's needed — never re-implement scaling/encoding/transform logic inline in trainer or UI code.

## Model Pipeline

```
Dataset → Task Selection → Model Recommendation → Model Factory → Trainer → Metrics → Visualization → Save Model → Inference
```

### Supported tasks

- **CSV**: Classification, Regression, Clustering, Time Series Forecasting
- **Images**: Image Classification
- **Future** (don't implement unless explicitly asked): Object Detection, Segmentation

### Classical ML (scikit-learn)

- Classification: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes
- Regression: Linear Regression, Ridge, Lasso
- Clustering: K-Means, DBSCAN, Hierarchical Clustering

### Deep Learning (PyTorch / TorchVision)

- CNN, RNN, LSTM, GRU, ResNet, EfficientNet
- Future (don't implement unless explicitly asked): Vision Transformer, Transformer, LLM

New architectures/algorithms are added through the **model factory**, not scattered ad hoc across trainer or UI code — the factory is the single place that maps a task + model name to a constructed model instance.

### Trainer responsibilities (`trainer/`)

- Train models
- Validate models
- Save checkpoints
- Early stopping
- Return metrics

Keep training loop logic here only — no Streamlit imports, no UI state in this module.

### Metrics (`metrics/`)

| Task | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1 Score |
| Regression | MAE, MSE, RMSE, R² |
| Clustering | Silhouette Score |

### Visualization (`visualization/`)

Generate: Loss Curve, Accuracy Curve, Confusion Matrix, ROC Curve, Precision-Recall Curve — using Matplotlib/Seaborn. Visualization functions should accept data/metrics as arguments and return figures; they should not know about Streamlit or file I/O paths directly.

### Inference (`inference/`)

- Load trained models
- Accept new data
- Generate predictions

## Coding Standards

**Always**
- Use type hints wherever possible.
- Add docstrings to public functions.
- Use meaningful variable names.
- Keep functions small and single-responsibility.
- Prefer composition over duplication.
- Follow PEP 8.

**Never**
- Hardcode file paths — read from `config/`.
- Duplicate preprocessing logic — extend `dataset/preprocessing.py` instead.
- Mix UI code with training logic.
- Put business logic inside Streamlit pages (`ui/`) — pages should only call into `app/` modules and render results.

## Commit Style

One logical feature per commit. Good examples: "Add dataset loader", "Implement dataset validator", "Create model factory", "Add CNN architecture", "Implement training loop". Don't bundle unrelated changes into one commit.

## Long-Term Direction

Keep additions compatible with where the project is heading, so today's code doesn't need rework later: Classical ML, Deep Learning, Transfer Learning, AutoML, Explainable AI, Object Detection, Segmentation, Transformers, Vision Transformers, and LLMs. When a design choice could go either a narrow, one-off way or a way that generalizes toward this roadmap, prefer the more general one — but don't build speculative infrastructure for features that haven't been requested yet.