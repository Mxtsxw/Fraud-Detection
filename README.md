# Fraud Detection - End-to-End ML Project

> **Detecting financial fraud in 6.3M transactions using feature engineering, class-imbalance strategies, and ensemble modeling — achieving >99% PR-AUC on held-out data.**

**[Try the Live App](https://github.com/Mxtsxw/Fraud-Detection/)**

<img width="2371" height="948" alt="Frame 1" src="https://github.com/user-attachments/assets/e7369461-1853-47e9-b427-0d0ba69d50ae" />
<img width="2759" height="912" alt="image" src="https://github.com/user-attachments/assets/3a0a626f-e44e-444a-82a0-d90b431d5cbf" />


---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Executive Summary](#executive-summary)
3. [Project Structure](#project-structure)
4. [Part 1 — Exploratory Data Analysis](#part-1--exploratory-data-analysis)
5. [Part 2 — Data Processing & Feature Engineering](#part-2--data-processing--feature-engineering)
6. [Part 3 — Modeling & Evaluation](#part-3--modeling--evaluation)
7. [Final Results](#final-results)
8. [Tech Stack](#tech-stack)

---

## Problem Statement

Financial fraud causes billions in losses annually and can be difficult to detect: fraudulent transactions are extremely rare, behavioral patterns are subtle, and class imbalance makes naive models useless. This project builds a fraud detection pipeline on a real-world synthetic financial [dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) of **6.36 million transactions**, tackling the full lifecycle from raw data exploration to tuned model evaluation.

**Core challenges addressed:**
- Severe class imbalance (~0.13% fraud rate — 8,213 fraud cases out of 6.36M transactions)
- Non-linear decision boundaries requiring ensemble methods
- Dataset-specific anomalies (balance inconsistencies, zero-balance draining patterns)
- Computational efficiency at scale

---

## Executive Summary

This project demonstrates a rigorous, end-to-end machine learning workflow structured across three notebooks. Starting from raw data, the analysis uncovered that **fraud is exclusively confined to TRANSFER and CASH_OUT transaction types**, and that **98% of fraudulent transactions completely drain the origin account**. These structural insights directly informed targeted feature engineering.

A preprocessing pipeline with MLflow experiment tracking and DVC data versioning was built to ensure reproducibility. Feature assessment using Logistic Regression and Random Forest baselines identified the most predictive signals before committing to full model training.

The final models, **Random Forest** and **XGBoost**, were trained on a stratified 1M-row subset (justified by a learning curve plateau analysis) and evaluated on a fully held-out test set. Both models exceeded **99.6% PR-AUC, Precision, Recall, and F1**, with XGBoost being **~4× faster** to train while maintaining near-identical performance.

---

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── Fraud_Detection_EDA_complete.ipynb        # Exploratory Data Analysis
│   ├── Fraud_Detection_Processing.ipynb          # Feature Engineering & Selection
│   └── Fraud_Detection_Modeling.ipynb            # Model Training & Evaluation
├── data/
│   ├── raw/                                      # Original dataset (DVC tracked)
│   └── processed/                                # Engineered features (DVC tracked)
├── mlruns/                                       # MLflow experiment logs
└── dvc_store/                                    # DVC local remote
└── app.py                                        # Streamlit Dashboard
```

---

## Part 1 - Exploratory Data Analysis

**Notebook**: `Fraud_Detection_EDA_complete.ipynb` ([here](https://github.com/Mxtsxw/Fraud-Detection/blob/main/notebooks/Fraud_Detection_EDA_complete.ipynb))

### Dataset Overview

| Property | Value |
|---|---|
| Rows | 6,362,620 |
| Columns | 11 |
| Missing Values | None |
| Duplicate Entries | None |
| Fraud Rate | 0.129% (8,213 transactions) |

**Features**: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`

---

### Key Finding 1 — Extreme Class Imbalance

The target variable `isFraud` is severely imbalanced. A naive "predict never fraud" model would achieve 99.87% accuracy. This finding justifies the use of precision/recall/F1 and PR-AUC as evaluation metrics, stratified train/test splits, and class-weighted loss functions throughout the project.

---

### Key Finding 2 — Fraud is Confined to Two Transaction Types

Fraud exclusively occurs in **TRANSFER** and **CASH_OUT** transactions. PAYMENT, DEBIT, and CASH_IN have zero fraud cases. This constitutes a relevant categorical signal in the dataset.

| Transaction Type | Fraud Count | Fraud Rate | Total Transactions |
|---|---|---|---|
| TRANSFER | 4,097 | 0.77% | 532,909 |
| CASH_OUT | 4,116 | 0.18% | 2,237,500 |
| CASH_IN | 0 | 0.00% | 1,399,284 |
| PAYMENT | 0 | 0.00% | 2,151,495 |
| DEBIT | 0 | 0.00% | 41,432 |

---

### Key Finding 3 — Account Draining Pattern

Fraudulent transactions almost always completely drain the origin account, a highly specific behavioral signature:

| Pattern | Non-Fraud | Fraud |
|---|---|---|
| `newbalanceOrig == 0` | 56.7% | **98.1%** |
| `oldbalanceOrg ≈ amount` (exact drain) | ~0.0% | **97.8%** |

---

### Key Finding 4 — Balance Inconsistency Anomalies

A structural dataset anomaly was identified: many transactions show discrepancies between expected and actual balance updates (`errorBalanceOrig`, `errorBalanceDest`). These errors are substantially more pronounced in fraud cases, particularly `errorBalanceDest`, where the destination balance is often never updated despite funds being transferred.

---

### Key Finding 5 — Fraud Targets Only Customer Accounts

Account names starting with `C` are customer accounts; `M` are merchant accounts. Fraud exclusively flows to customer (C) destinations — never to merchant accounts. This clean binary signal was directly encoded as a feature.

---

### Key Finding 6 — Temporal Patterns

The `step` variable might represent ~1-hour intervals over a ~30-day period (743 total steps). Fraud counts and rates show some periodicity, suggesting daily cycles. However, transaction volume decreases in the second half of the period.

<img width="1184" height="684" alt="image" src="https://github.com/user-attachments/assets/db7a8468-2fd0-437e-b275-d34294d3de64" />

---

### Key Finding 7 — Correlation Structure

Balance variables show high internal correlation (`oldbalanceOrg`↔`newbalanceOrig`, `oldbalanceDest`↔`newbalanceDest`), suggesting that balance difference/error features may be more informative than the raw before/after pairs individually.

<img height="300" alt="image" src="https://github.com/user-attachments/assets/b5c79252-5bcf-4e50-adfa-a282acf08ccf" />

---

## Part 2 — Data Processing & Feature Engineering

**Notebook**: `Fraud_Detection_Processing.ipynb` ([here](https://github.com/Mxtsxw/Fraud-Detection/blob/main/notebooks/Fraud_Detection_Processing.ipynb))

All engineering decisions are directly grounded in EDA findings and documented with explicit hypotheses, actions, and considerations. The pipeline uses **MLflow** for experiment tracking and **DVC** for data versioning.

### Feature Engineering

| Feature | Rationale | Status |
|---|---|---|
| `logAmount` | Compresses right-skewed amount (range 0–92M) | ✅ Kept |
| `errorBalanceOrig` | Balance inconsistency at origin — strong fraud signal | ✅ Kept |
| `errorBalanceDest` | Balance inconsistency at destination — strongest EDA signal | ✅ Kept |
| `destTypeC` | Fraud exclusively targets C accounts — binary, noise-free signal | ✅ Kept |
| `isOrigDrained` | 98.1% of fraud drains origin account — high-precision flag | ✅ Kept |
| `hourSin` / `hourCos` | Cyclic time encoding of `step % 24` | ❌ Dropped (low consensus score) |
| `step` (raw) | Temporal signal | ❌ Dropped (low consensus score) |
| `nameOrig` / `nameDest` | High-cardinality IDs, no generalizable signal | ❌ Excluded |
| `isFlaggedFraud` | Leaky rule-based flag, captures only 0.19% of fraud | ❌ Excluded |

**Final feature count**: 10 features — `type`, `logAmount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `errorBalanceOrig`, `errorBalanceDest`, `destTypeC`, `isOrigDrained`

### Train / Test Split

An 80/20 stratified split was used to preserve the fraud rate across both sets:

| Split | Rows | Fraud Count | Fraud Rate |
|---|---|---|---|
| Train | 4,453,834 | 5,749 | 0.129% |
| Test | 1,908,786 | 2,464 | 0.129% |

The test set was held out and not touched until final model evaluation.


### Class Imbalance Strategy

`class_weight='balanced'` was chosen. This weights each class inversely proportional to its frequency without modifying the training distribution.

---

### Feature Assessment — Logistic Regression Baseline

Logistic Regression was used as a baseline to assess linear signal in the feature set. Results on a 500K-row sample:

| Metric | Score |
|---|---|
| ROC-AUC | 0.9928 |
| PR-AUC (Avg. Precision) | 0.7044 |
| Recall | 0.9891 |
| Precision | 0.0294 |
| F1 | 0.0570 |

High recall but very low precision confirmed that the feature space has strong linear signal but that the decision boundary is non-linear → pointing toward tree-based models.


<img width="884" height="684" alt="image" src="https://github.com/user-attachments/assets/3dbf9955-ee89-456c-b6ec-0a1b7b718dff" />


---

### Feature Assessment — Decision Tree

Random Forest feature importances showed `errorBalanceOrig`, `isOrigDrained`, `oldbalanceOrg`, `newbalanceOrig`, and `logAmount` as the top predictors, which is consistent with EDA findings.


<img width="884" height="684" alt="image" src="https://github.com/user-attachments/assets/035b41db-d187-46ed-be45-1f72fa511e93" />


---

### Feature Selection — Consensus Ranking

Features were ranked by normalizing LR coefficients and RF importances to [0,1] and averaging. Features with consensus score < 0.05 were dropped:

| Dropped Feature | LR Score | RF Score | Consensus |
|---|---|---|---|
| `step` | 0.005 | 0.088 | 0.047 |
| `hourSin` | 0.016 | 0.012 | 0.014 |
| `hourCos` | 0.001 | 0.003 | 0.002 |


<img width="884" height="684" alt="image" src="https://github.com/user-attachments/assets/56d8f56b-651c-4cd4-81c5-61c769de0782" />


---

### Learning Curve Analysis

A learning curve on Average Precision showed performance plateaus around **1,000,000 training rows**, justifying use of a stratified 1M-row subset for all tuning experiments. This reduces training time by ~4× with negligible performance loss.

<img width="791" height="587" alt="image" src="https://github.com/user-attachments/assets/0820349d-cae6-49d5-a9c0-64e6326a0ce8" />


---

## Part 3 — Modeling & Evaluation

**Notebook**: `Fraud_Detection_Modeling.ipynb` ([here](https://github.com/Mxtsxw/Fraud-Detection/blob/main/notebooks/Fraud_Detection_Modeling.ipynb))

---

### Hyperparameter Tuning

Grid search with `StratifiedKFold(n=3)` on the 1M-row subset, scored on `average_precision` (PR-AUC). All runs logged to MLflow.

**Random Forest — Best Configuration:**

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_leaf` | 50 |
| **Best CV PR-AUC** | **0.9977** |
| Grid search time | 510.7s |

**XGBoost — Best Configuration:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 773.6 |
| **Best CV PR-AUC** | **0.9959** |
| Grid search time | 55.4s |

---

### Final Evaluation on Held-Out Test Set

| Model | PR-AUC | Precision | Recall | F1 | Train Time |
|---|---|---|---|---|---|
| **Random Forest** | **0.9972** | **0.9996** | 0.9963 | **0.9980** | 12.9s |
| **XGBoost** | 0.9969 | 0.9903 | **0.9963** | 0.9933 | **3.4s** |

Both models dramatically outperform the Logistic Regression baseline, confirming that non-linear ensemble methods are necessary for this problem.

<img width="1173" height="873" alt="image" src="https://github.com/user-attachments/assets/29861db2-9cb0-4cb5-b87e-64969c1d85ba" />


---

### Training Speed Analysis

| Model | Train Time | Throughput |
|---|---|---|
| Random Forest | 12.9s | 77,262 rows/s |
| XGBoost | **3.4s** | **294,494 rows/s** |

XGBoost is ~**4× faster** than Random Forest, attributed to its `hist` tree method's approximate binning vs. RF's independent parallel tree construction.

<img width="1626" height="759" alt="speed_comparison" src="https://github.com/user-attachments/assets/4d6df1e0-62b7-4fb4-b5dd-047542505e35" />

---

## Final Results

| Model | PR-AUC | Precision | Recall | F1 | Train Time | Recommendation |
|---|---|---|---|---|---|---|
| Random Forest | 0.9972 | **0.9996** | 0.9963 | **0.9980** | 12.9s | Best for precision-critical deployments |
| XGBoost | 0.9969 | 0.9903 | 0.9963 | 0.9933 | **3.4s** | Best for latency-sensitive or high-throughput systems |

**Both models are production-viable.** The choice between them depends on deployment constraints: Random Forest offers marginally fewer false positives, while XGBoost provides 4× faster training and inference which is a meaningful advantage for real-time fraud scoring and re-training strategies.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Experiment Tracking | `mlflow` |
| Data Versioning | `dvc` |
| Environment | Python, Jupyter Notebooks |
