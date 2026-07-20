from pathlib import Path
from config import REPO_ROOT
from universal_utils import get_cur_prefix_dir
from scenarios.attr_inference.utils import load_yaml
from utils import load_labels_df, build_derived_features, align_subjects
from train_eval import evaluate_all_attributes

CONFIG_PATH = REPO_ROOT / "scenarios" / "attr_inference" / "attr_inference.yaml"
OUT_DIR = REPO_ROOT / "results" / "02_attr_inference_supervised_ml"

if __name__ == "__main__":
    config = load_yaml(CONFIG_PATH)
    data_path = get_cur_prefix_dir(REPO_ROOT / config["data_root"], "version")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, subject_ids = build_derived_features(data_path)
    labels_df = load_labels_df(data_path)
    X, labels_df = align_subjects(X, subject_ids, labels_df)

    evaluate_all_attributes(
        X              = X,
        labels_df      = labels_df,
        target_info    = config["target_info"],
        condition_name = "derived_biomarkers",
        output_path    = OUT_DIR / "derived_biomarkers.json",
    )