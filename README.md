# Fruit Classification CNN

**SKKU Introduction to Deep Learning — Assignment 2**

257-class fruit image classification using a custom-designed CNN built with PyTorch.

## Task Overview

- **Dataset**: Fruits-360 (re-split, anonymized)
  - Train: 126,040 images
  - Validation: 18,023 images
  - Test: 36,016 images
- **Classes**: 257 fruit categories (label 0–256)
- **Metric**: Categorization Accuracy
- **Platform**: Kaggle competition

## Constraints

- PyTorch only
- Custom CNN architecture (no pretrained models)
- 1 epoch training

## Project Structure

```
├── .cursor/rules/          # Cursor AI rules
├── SimpleBaseline_CNN_kaggle-2.py  # Baseline code
└── README.md
```

## Requirements

```bash
pip install torch torchvision tqdm pandas numpy pillow
```
