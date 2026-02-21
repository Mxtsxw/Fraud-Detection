import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate

def evaluate_pipeline(pipeline, X, y, cv, label="model", use_subset=False, subset_size=500_000):
    """
    Run k-fold stratified CV and return a dict of mean scores.
    Uses scoring metrics appropriate for imbalanced classification.
    """

    # 1. Handling the scale
    if use_subset and len(X) > subset_size:
        X_eval, _, y_eval, _ = train_test_split(
            X, y, train_size=subset_size, stratify=y, random_state=42
        )
        print(f"Sampling {subset_size} rows for fast evaluation...")
    else:
        X_eval, y_eval = X, y

    scoring = {
        'roc_auc'    : 'roc_auc',
        'avg_prec'   : 'average_precision',
        'f1_fraud'   : 'f1',           # default pos_label=1 = fraud
        'precision'  : 'precision',
        'recall'     : 'recall',
    }
    results = cross_validate(pipeline, X_eval, y_eval, cv=cv,
                             scoring=scoring, n_jobs=-1)
    summary = {k: results[f'test_{k}'].mean() for k in scoring}
    print(f"\n── {label} ──")
    for k, v in summary.items():
        print(f"  {k:<12}: {v:.4f}")

    return summary


def sample_stratified_subset(X, y, n=500_000, random_seed=42):
    # Calculate fraction needed
    frac = n / len(X)
    # Use train_test_split with stratify
    X_subset, _, y_subset, _ = train_test_split(
        X, y,
        train_size=frac,
        stratify=y,
        random_state=random_seed
    )
    return X_subset, y_subset


