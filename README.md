# Diabetic Retinopathy Classification from Retinal Fundus Images

Applied machine-learning and deep-learning project for automated diabetic retinopathy assessment from retinal fundus images.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vmKS1l1UkW9euSb8uXdvBaXPhatuzgJJ?usp=sharing)

## Overview

Diabetic retinopathy is a diabetes-related eye disease that can cause irreversible vision loss if it is not detected early. This project explores multiple approaches for learning from a large retinal-image dataset, ranging from classical machine learning on learned image representations to end-to-end deep learning.

The project uses roughly **35,000 retinal fundus images** and evaluates both conventional ML and neural-network approaches.

## What I built

- Image loading and preprocessing pipelines for retinal fundus images.
- Custom PyTorch `Dataset` implementations for binary and severity-label experiments.
- Deep-feature extraction followed by **XGBoost** classification.
- End-to-end transfer learning / fine-tuning with **EfficientNetV2-S**.
- Evaluation with classification metrics suited to medical-image prediction.
- Notebook-based experimentation designed to compare modeling approaches rather than a single fixed architecture.

## Results

| Approach | Reported result |
| --- | ---: |
| Deep features + XGBoost | **Accuracy: 0.7492** |
| Deep features + XGBoost | **Quadratic Kappa: 0.6853** |
| EfficientNetV2-S | **AUC: 0.9018** |

These results reflect the best reported project runs and are included here to make the main outcomes visible without requiring readers to inspect the notebooks.

## Repository structure

```text
.
├── Project.ipynb              # Main experimentation notebook
├── Project_suggestion.ipynb   # Additional / extended experimentation
├── DRDataset.py               # Custom PyTorch dataset
├── DR_HR_Dataset.py           # Dataset supporting binary/severity labels
└── .gitignore                 # Excludes datasets and local artifacts
```

## Run the project

The easiest way to inspect the work is through the Colab notebook linked above. The raw image data is intentionally not stored in this repository, so local paths or mounted Drive paths may need to be adjusted before running the notebooks.

For local use, create a Python environment with the packages used by the notebooks, including PyTorch, pandas, NumPy, scikit-learn, XGBoost, Pillow, and the relevant visualization utilities.

## Skills demonstrated

`Python` · `PyTorch` · `Computer Vision` · `Deep Learning` · `Transfer Learning` · `XGBoost` · `scikit-learn` · `pandas` · `Medical Imaging`

## Notes

This was developed as an academic data-science project. The repository is intended to present the modeling workflow, experiments, and results; the source dataset is not redistributed here.
