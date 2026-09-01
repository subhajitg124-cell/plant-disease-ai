# Plant Disease Vision Model — Baseline Performance Report

## Summary & Overview
- **Model Architecture**: `PlantDiseaseCNN` (4-Block ConvNet + 128-dim Visual Embedding + 38-class Head)
- **Model Checkpoint**: `models/plant_disease_cnn.pth`
- **Total Samples Evaluated**: 38

## Core Metrics

| Metric | Score |
| :--- | :--- |
| **Top-1 Accuracy** | `0.00%` |
| **Top-3 Accuracy** | `7.89%` |
| **Top-5 Accuracy** | `13.16%` |
| **Macro Precision** | `0.0000` |
| **Macro Recall** | `0.0000` |
| **Macro F1-Score** | `0.0000` |
| **Weighted F1-Score** | `0.0000` |

## Evaluation Status
- **Preprocessing Pipeline**: Active (`224x224 RGB`, ImageNet normalization)
- **Input Validation**: Active (`PredictionStatus.NOT_A_PLANT` foliage check)
- **Visual Embeddings**: 128-dimensional L2-normalized feature vectors
