"""
scripts/supervised_ml/train_eval.py

Cross-validated supervised ML evaluation pipeline.
Establishes the supervised upper bound for each attribute, used to verify
that signal exists in the data before interpreting LLM null results.

Usage (called by run_raw.py and run_derived.py, not directly):
    from train_eval import evaluate_all_attributes
    results = evaluate_all_attributes(X, labels_df, condition_name, output_path)
"""

import json
import time
import numpy as np
from pathlib import Path
from collections import Counter

from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRegressor

from utils import (
    load_labels_df,        # () -> pd.DataFrame with columns: subject_id, formal_status, age, gender, education
    align_subjects,        # (X_dict, labels_df) -> (X_np, labels_df_aligned) ensuring row correspondence
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FOLDS = 5
RANDOM_STATE = 42

FORMAL_STATUS_CHOICES = [
    "HEALTHY", "MDD", "ADHD", "OCD", "INSOMNIA",
    "TINNITUS", "CHRONIC PAIN", "ADD", "TINNITUS/MDD",
]

XGBC_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

XGBR_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Per-attribute CV routines
# ---------------------------------------------------------------------------

def _cv_formal_status(X: np.ndarray, y_raw: np.ndarray) -> dict:
    """
    Multiclass classification. Stratified K-fold to preserve class balance.
    Metrics: top-1 accuracy, macro-F1, and baseline majority-class accuracy.
    Returns per-fold arrays and aggregated results.
    """
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accs, fold_f1s = [], []
    oof_preds = np.empty(len(y), dtype=object)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(num_class=len(classes), **XGBC_PARAMS)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        oof_preds[val_idx] = le.inverse_transform(preds)

        acc = np.mean(preds == y_val)
        fold_accs.append(acc)

        # Per-class F1 then macro average
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)

    # Naive baseline: always predict majority class
    majority = Counter(y).most_common(1)[0][0]
    baseline_acc = np.mean(y == majority)

    return {
        "type": "categorical",
        "n": len(y),
        "cv_top1_acc": float(np.mean(fold_accs)),
        "cv_top1_acc_std": float(np.std(fold_accs)),
        "cv_macro_f1": float(np.mean(fold_f1s)),
        "cv_macro_f1_std": float(np.std(fold_f1s)),
        "baseline_top1_acc": float(baseline_acc),
        "fold_accs": [float(a) for a in fold_accs],
        "class_distribution": {k: int(v) for k, v in Counter(y_raw).items()},
    }


def _cv_gender(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Binary classification (0/1). Stratified K-fold.
    Metrics: accuracy and F1. Gender is already integer-encoded.
    """
    y = y.astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accs, fold_f1s = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(num_class=2, **XGBC_PARAMS)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        fold_accs.append(float(np.mean(preds == y_val)))
        fold_f1s.append(float(f1_score(y_val, preds, average='binary', zero_division=0)))

    majority = int(Counter(y).most_common(1)[0][0])
    baseline_acc = float(np.mean(y == majority))

    return {
        "type": "binary",
        "n": len(y),
        "cv_top1_acc": float(np.mean(fold_accs)),
        "cv_top1_acc_std": float(np.std(fold_accs)),
        "cv_f1": float(np.mean(fold_f1s)),
        "cv_f1_std": float(np.std(fold_f1s)),
        "baseline_top1_acc": baseline_acc,
        "fold_accs": [float(a) for a in fold_accs],
        "class_distribution": {str(k): int(v) for k, v in Counter(y).items()},
    }


def _cv_age(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Continuous regression. Standard K-fold (not stratified; age is continuous).
    Metrics: MAE, Pearson r. Baseline is predicting the training-fold mean.
    """
    y = y.astype(float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_maes, fold_rs, baseline_maes = [], [], []
    oof_preds = np.empty(len(y))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**XGBR_PARAMS)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        fold_maes.append(float(np.mean(np.abs(preds - y_val))))

        # Baseline: predict training mean
        train_mean = float(np.mean(y_tr))
        baseline_maes.append(float(np.mean(np.abs(train_mean - y_val))))

        if np.std(preds) > 0 and np.std(y_val) > 0:
            r, _ = pearsonr(preds, y_val)
            fold_rs.append(float(r))

    # OOF Pearson r across all subjects
    oof_r, oof_p = pearsonr(oof_preds, y)

    return {
        "type": "continuous",
        "n": len(y),
        "cv_mae": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
        "cv_pearson_r": float(np.mean(fold_rs)) if fold_rs else None,
        "oof_pearson_r": float(oof_r),
        "oof_pearson_p": float(oof_p),
        "baseline_mae": float(np.mean(baseline_maes)),
        "fold_maes": [float(m) for m in fold_maes],
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }


def _cv_education(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Ordinal regression treated as regression (standard for education years).
    Metrics: MAE, Spearman rho (rank correlation, appropriate for ordinal).
    """
    y = y.astype(float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_maes, fold_rhos, baseline_maes = [], [], []
    oof_preds = np.empty(len(y))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**XGBR_PARAMS)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        fold_maes.append(float(np.mean(np.abs(preds - y_val))))

        train_mean = float(np.mean(y_tr))
        baseline_maes.append(float(np.mean(np.abs(train_mean - y_val))))

        if np.std(preds) > 0:
            rho, _ = spearmanr(preds, y_val)
            fold_rhos.append(float(rho))

    oof_rho, oof_p = spearmanr(oof_preds, y)

    return {
        "type": "ordinal",
        "n": len(y),
        "cv_mae": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
        "cv_spearman_rho": float(np.mean(fold_rhos)) if fold_rhos else None,
        "oof_spearman_rho": float(oof_rho),
        "oof_spearman_p": float(oof_p),
        "baseline_mae": float(np.mean(baseline_maes)),
        "fold_maes": [float(m) for m in fold_maes],
        "y_distribution": {str(int(k)): int(v) for k, v in Counter(y.astype(int)).items()},
    }

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_all_attributes(
    X: np.ndarray,
    subject_ids: list[str],
    condition_name: str,
    output_path: str | Path,
) -> dict:
    """
    Run cross-validated evaluation for all four attributes.

    Args:
        X:              Feature matrix, shape (n_subjects, n_features).
                        Row order must match subject_ids.
        subject_ids:    List of subject ID strings corresponding to X rows.
        condition_name: e.g. "raw_spectral" or "derived_biomarkers".
                        Written into output meta.
        output_path:    Where to write the results JSON.

    Returns:
        Full results dict (also written to output_path).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ground truth labels and align to X row order
    labels_df = load_labels_df()
    X, labels_df = align_subjects(X, subject_ids, labels_df)

    n = len(labels_df)
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    print(f"Subjects: {n}  |  Features: {X.shape[1]}")
    print(f"{'='*60}")

    results = {
        "meta": {
            "condition": condition_name,
            "n_subjects": n,
            "n_features": int(X.shape[1]),
            "n_folds": N_FOLDS,
            "model": "XGBoost",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }

    # --- formal_status ---
    print("\n[1/4] formal_status (multiclass classification)...")
    y_fs = labels_df["formal_status"].values
    results["formal_status"] = _cv_formal_status(X, y_fs)
    _print_summary("formal_status", results["formal_status"])

    # --- gender ---
    print("\n[2/4] gender (binary classification)...")
    y_g = labels_df["gender"].values
    results["gender"] = _cv_gender(X, y_g)
    _print_summary("gender", results["gender"])

    # --- age ---
    print("\n[3/4] age (continuous regression)...")
    y_a = labels_df["age"].values
    results["age"] = _cv_age(X, y_a)
    _print_summary("age", results["age"])

    # --- education ---
    print("\n[4/4] education (ordinal regression)...")
    y_e = labels_df["education"].values
    results["education"] = _cv_education(X, y_e)
    _print_summary("education", results["education"])

    # Write output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")

    return results


def _print_summary(attr: str, result: dict) -> None:
    t = result["type"]
    baseline_key = "baseline_top1_acc" if t in ("categorical", "binary") else "baseline_mae"
    model_key = "cv_top1_acc" if t in ("categorical", "binary") else "cv_mae"
    baseline = result[baseline_key]
    model = result[model_key]

    if t in ("categorical", "binary"):
        delta = model - baseline
        direction = "▲" if delta > 0 else "▼"
        print(f"  {attr}: acc={model:.3f} vs baseline={baseline:.3f}  {direction}{abs(delta):.3f}")
    else:
        delta = model - baseline
        direction = "▼" if delta < 0 else "▲"  # lower MAE is better
        print(f"  {attr}: MAE={model:.3f} vs baseline={baseline:.3f}  {direction}{abs(delta):.3f}")