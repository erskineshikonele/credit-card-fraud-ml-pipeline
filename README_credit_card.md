# 💳 Credit Card Fraud Detection — ML Pipeline

**Author:** Erskine Brister Shikonele | Student No: 218182406  
**Module:** EIARI4A — Artificial Intelligence  
**Institution:** Vaal University of Technology

---

## 📋 Overview

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The dataset is highly imbalanced — only 0.17% of 284,807 transactions are fraudulent — making this a real-world class imbalance problem. The project applies SMOTE oversampling and compares six classification models to find the best-performing fraud detector.

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `creditcard.csv` |
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | 30 (V1–V28 PCA components + Time + Amount) |
| Target | `Class` (0 = Legitimate, 1 = Fraud) |

> The features V1–V28 are PCA-transformed to protect cardholder privacy.

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of decision trees |
| AdaBoost | Adaptive boosting |
| CatBoost | Gradient boosting for categorical data |
| XGBoost | Extreme gradient boosting |
| LightGBM | Light gradient boosting machine |

---

## ⚙️ Pipeline

```
Raw CSV → Exploratory Data Analysis
       → Train / Test Split (80/20)
       → SMOTE oversampling on training set
       → StandardScaler normalisation
       → Model training & evaluation
       → ROC-AUC, Precision, Recall, F1 comparison
       → Confusion matrices
       → Best model selection
```

---

## 📈 Evaluation Metrics

Since accuracy is misleading on imbalanced datasets, models were evaluated on:

- **ROC-AUC Score** — overall discrimination ability
- **Precision** — of predicted fraud, how many were actually fraud
- **Recall** — of actual fraud, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visual breakdown of predictions

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | ML models, metrics, preprocessing |
| Imbalanced-learn | SMOTE oversampling |
| XGBoost | XGBoost classifier |
| LightGBM | LightGBM classifier |
| CatBoost | CatBoost classifier |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualisation |
| Jupyter Notebook | Development environment |

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/erskineshikonele/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib seaborn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook credit-card-fraud-detection-predictive-models.ipynb
```

**4. Run all cells** — results, plots, and model comparisons appear inline.

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
├── credit-card-fraud-detection-predictive-models.ipynb   # Main notebook
├── creditcard.csv                                         # Dataset
├── Credit_Card_Fraud_Detection_Presentation.pdf          # Project presentation
└── README.md
```

---

## ⚠️ Disclaimer

This project is for academic purposes only. The dataset is sourced from Kaggle (ULB Machine Learning Group) and used strictly for educational research as part of the EIARI4A Artificial Intelligence module at Vaal University of Technology.

---

## 📄 Licence

MIT — free to use with attribution.
