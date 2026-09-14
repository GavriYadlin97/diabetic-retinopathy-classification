# Diabetic Retinopathy Classification from Retinal Fundus Images

Applied machine-learning and deep-learning project for automated diabetic retinopathy assessment from retinal fundus images.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vmKS1l1UkW9euSb8uXdvBaXPhatuzgJJ?usp=sharing)

## Overview

Diabetic retinopathy is a diabetes-related eye disease that can cause irreversible vision loss if it is not detected early. This project explores multiple approaches for learning from a large retinal-image dataset, ranging from classical machine learning on learned image representations to end-to-end deep learning.

The project uses roughly **35,000 retinal fundus images** and evaluates both conventional ML and neural-network approaches while addressing substantial class imbalance.

## What I built

- Image loading, preprocessing, and augmentation pipelines for retinal fundus images.
- Custom PyTorch `Dataset` implementations for binary and severity-label experiments.
- Deep-feature extraction followed by **XGBoost** classification.
- End-to-end transfer learning / fine-tuning with **EfficientNetV2-S**.
- A **two-stage Vision Transformer (ViT) classification pipeline** for diabetic-retinopathy severity prediction.
- Evaluation using accuracy, quadratic weighted kappa (QWK), F1, precision, recall, and AUC.
- Notebook-based experimentation comparing multiple modeling strategies rather than a single fixed architecture.

## Results

The strongest overall result came from the **two-stage ViT pipeline**.

| Approach | Accuracy | QWK | F1 | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Two-stage ViT (validation)** | **0.88** | **0.864** | **0.754** | **0.779** | **0.747** | **0.842** |
| Deep features + XGBoost | **0.7492** | **0.6853** | — | — | — | — |
| EfficientNetV2-S | — | — | — | — | — | **0.9018** |

The ViT pipeline produced the best overall severity-classification result, with **0.864 quadratic weighted kappa** and **88% validation accuracy**. The XGBoost and EfficientNet experiments provide complementary baselines showing the progression from feature-based ML to end-to-end deep learning and transformer-based modeling.

## Repository structure

```text
.
├── Project.ipynb              # Main experimentation notebook
├── Project_suggestion.ipynb   # Extended experiments and modeling
├── DRDataset.py               # Custom PyTorch dataset
├── DR_HR_Dataset.py           # Dataset supporting binary/severity labels
└── .gitignore                 # Excludes datasets and local artifacts
```

## Run the project

The easiest way to inspect the work is through the Colab notebook linked above. The raw image data is intentionally not stored in this repository, so local paths or mounted Drive paths may need to be adjusted before running the notebooks.

For local use, create a Python environment with the packages used by the notebooks, including PyTorch, pandas, NumPy, scikit-learn, XGBoost, Pillow, and the relevant visualization utilities.

## Skills demonstrated

`Python` · `PyTorch` · `Computer Vision` · `Vision Transformers` · `Deep Learning` · `Transfer Learning` · `XGBoost` · `scikit-learn` · `pandas` · `Medical Imaging` · `Imbalanced Classification`

## Notes

This was developed as an academic data-science project. The repository is intended to present the modeling workflow, experiments, and results; the source dataset is not redistributed here.
