import json
import numpy as np
import pandas as pd
from scenarios.attr_inference.utils import load_json

CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CONDITIONS = ["EC", "EO"]

# Flat column names: EC_Fp1_delta, EC_Fp1_theta, ..., EO_O2_gamma
FEATURE_COLUMNS = [
    f"{cond}_{ch}_{band}"
    for cond in CONDITIONS
    for ch in CHANNELS
    for band in BANDS
]  # length 260

def load_labels_df(data_path):
    """
    Load labels.json and return a tidy DataFrame with only the four
    target attributes we score on.

    Returns:
        DataFrame with columns: subject_id, formal_status, age, gender, education
        subject_id is a plain column (not the index) to make align_subjects simpler.
    """
    labels = load_json(data_path / "labels.json")

    rows = []
    for subject_id, fields in labels.items():
        rows.append({
            "subject_id":    subject_id,
            "formal_status": fields["formal_status"],
            "age":           fields["age"],
            "gender":        int(fields["gender"]),
            "education":     fields["education"],
        })

    return pd.DataFrame(rows)

def align_subjects(
    X: np.ndarray,
    subject_ids: list[str],
    labels_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Ensure X rows and labels_df rows correspond to the same subjects
    in the same order.

    subject_ids is the authoritative ordering (matches X rows).
    labels_df may contain subjects not in subject_ids (e.g. ones that were filtered out during feature extraction), which are dropped.

    Returns: (X, labels_df_aligned): same X, labels filtered and reordered to match subject_ids exactly.
    """
    labels_indexed = labels_df.set_index("subject_id")

    missing = [sid for sid in subject_ids if sid not in labels_indexed.index]
    if missing:
        raise ValueError(
            f"{len(missing)} subject(s) in feature matrix have no label entry: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    labels_aligned = labels_indexed.loc[subject_ids].reset_index()
    assert len(labels_aligned) == len(subject_ids) == len(X), \
        "Row count mismatch after alignment — something went wrong."

    return X, labels_aligned

def build_raw_features(data_path) -> tuple[np.ndarray, list[str]]:
    """
    Load records.json and flatten each subject's EC/EO spectral values into a fixed-length feature vector following FEATURE_COLUMNS ordering.

    Returns:
        X: np.ndarray of shape (n_subjects, 260)
        subject_ids: list of subject ID strings, same row order as X
    """
    records = load_json(data_path / "records.json")

    subject_ids, rows = [], []

    for subject_id, conditions in records.items():
        row = []
        for cond in CONDITIONS:        # EC then EO
            for ch in CHANNELS:        # Fp1 ... O2
                for band in BANDS:     # delta ... gamma
                    row.append(conditions[cond][ch][band])
        rows.append(row)
        subject_ids.append(subject_id)

    X = np.array(rows, dtype=np.float32)
    return X, subject_ids

def build_derived_features(data_path):
    pass