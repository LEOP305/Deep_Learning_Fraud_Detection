# Credit Card Fraud Detection Using Deep Learning

A deep learning project for detecting fraudulent credit card transactions using an Artificial Neural Network (ANN).

> Built as part of Tuwaiq Academy's Data Science and AI Bootcamp.

## Overview

Credit card fraud detection is a highly imbalanced binary classification problem, where fraudulent transactions represent only a very small proportion of all transactions.

This project develops a neural-network-based fraud detection pipeline and focuses on handling the challenges associated with severe class imbalance.

The project covers:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Train/validation/test splitting
- Feature scaling
- Class-weighted training
- Artificial Neural Network (ANN) development
- L2 regularization and Dropout
- Early stopping
- Hyperparameter tuning
- Classification threshold optimization
- Final model evaluation
- Model and preprocessing artifact serialization

---

## Dataset

The project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains:

- **284,807 transactions**
- **30 input features**
- **1 binary target variable (`Class`)**

The target variable is:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

The dataset is highly imbalanced:

| Class | Transactions |
|---|---:|
| Legitimate | 284,315 |
| Fraudulent | 492 |

Fraudulent transactions therefore represent approximately **0.17%** of the dataset.

The features `V1`–`V28` are anonymized numerical features, while `Time` and `Amount` represent the transaction time and transaction amount.

---

## Project Structure

```text
fraud-detection/
│
│
├── data/
│   └── cleaned_data.parquet
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
│
├── artifacts/
│   ├── fraud_detection_model.keras
│   ├── scaler.pkl
│   ├── threshold.json
│   ├── model_config.json
│   ├── final_metrics.csv
│   ├── hyperparameter_results.csv
│   └── training_history.csv
│
├── app.py
│
├── README.md
├── .gitignore
└── requirements.txt
