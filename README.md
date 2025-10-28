# transient-detection
SVM-based classification for astronomical transient vs non-transient detection

# Transient vs Non-Transient Detection

Machine learning project for classifying astronomical events as transient or non-transient using Support Vector Machine (SVM).

## Problem Statement
Binary classification of astronomical events:
- **Transient (1)**: Temporary astronomical events
- **Non-Transient (0)**: Constant astronomical objects

## Dataset
- Training data with multiple astronomical features
- Binary labels: 0 (non-transient), 1 (transient)
- Evaluation metric: Accuracy

## Model Performance
- **Algorithm**: Support Vector Machine (SVM with RBF kernel)
- **Accuracy**: 98%
- **Preprocessing**: StandardScaler for feature normalization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ksrisaitej/transient-detection.git
cd transient-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

