from scenarios.attr_inference.utils import load_json, load_yaml, process_categorical, process_continuous, majority_class_acc
from config import REPO_ROOT
from universal_utils import get_cur_prefix_dir

# continuous
from scipy.stats import pearsonr, spearmanr
import numpy as np

import json

if __name__ == "__main__":
    config_path = REPO_ROOT / "scenarios" / "attr_inference" / "attr_inference.yaml"
    config = load_yaml(config_path)

    
    data_path = get_cur_prefix_dir(REPO_ROOT / config["data_root"], "version")

    labels = load_json(data_path / "labels.json")

    #results_dir = get_cur_prefix_dir(REPO_ROOT / config["output_dir"], "run")
    # alternative hard-coded results_dir below:
    results_dir = REPO_ROOT / config["output_dir"] / "run_003"
    
    results = load_json(results_dir / "results.json")

    target_info = {t["name"]: t for t in config["target_info"]}
    target_names = [t["name"] for t in config["target_info"]]
    topk = {name: {"correct-1": 0.0, "correct-2": 0.0, "correct-3": 0.0, "total": 0.0} for name in target_names}
    mrr_data = {name: [] for name in target_names}
    f05_data = {name: [] for name in target_names}
    continuous_data = {name: [] for name in target_names}

    for subj_id, inference in results.items():
        if subj_id == "meta":
            continue
        if labels.get(subj_id) is None:
            print(f"Error with {subj_id}, does not exist in labels.")
            continue
        
        label = labels[subj_id]

        for attr_name, attr_inference in inference["estimates"].items():
            attr_type = target_info[attr_name]["type"]
            label_val = label[attr_name]
            if attr_type in ("categorical", "binary"):
                process_categorical(attr_name, attr_inference, label_val, topk, mrr_data, f05_data, target_info, subj_id)
            elif attr_type in ("continuous", "ordinal"):
                process_continuous(attr_name, attr_inference, label_val, continuous_data)

    # categorical
    for name in target_names:
        if target_info[name]["type"] not in ("categorical", "binary"):
            continue
        if topk[name]['total'] > 0:
            topk[name]['top1_acc'] = float(topk[name]['correct-1'] / topk[name]['total'])
        topk[name]['mrr'] = float(sum(mrr_data[name]) / len(mrr_data[name])) if mrr_data[name] else None

    # continuous / ordinal
    continuous_scores = {}
    for name in target_names:
        pairs = continuous_data[name]
        if not pairs:
            continue
        preds = [p for p, _ in pairs]
        actuals = [a for _, a in pairs]
        
        preds_arr = np.array(preds)
        actuals_arr = np.array(actuals)
        
        mae = np.mean(np.abs(preds_arr - actuals_arr))
        baseline_mae = np.mean(np.abs(actuals_arr - np.mean(actuals_arr)))
        
        continuous_scores[name] = {
            "mae": float(mae),
            "baseline_mae": float(baseline_mae),
            "n": len(pairs),
        }
        
        if target_info[name]["type"] == "continuous":
            r, p_r = pearsonr(preds_arr, actuals_arr)
            continuous_scores[name]["pearson_r"] = float(r)
            continuous_scores[name]["pearson_p"] = float(p_r)
        elif target_info[name]["type"] == "ordinal":
            rho, p_rho = spearmanr(preds_arr, actuals_arr)
            continuous_scores[name]["spearman_rho"] = float(rho) if not np.isnan(rho) else None
            continuous_scores[name]["spearman_p"] = float(p_rho) if not np.isnan(p_rho) else None

    # assemble scores
    scores = {"meta": {
        "model": config["model"],
        "run_dir": str(results_dir),
        "n_subjects": len(results) - 1,
    }}

    for name in target_names:
        t = target_info[name]["type"]
        if t in ("categorical", "binary"):
            scores[name] = {
                "type": t,
                "n": int(topk[name]["total"]),
                "top1_acc": topk[name].get("top1_acc", 0.0),
                "mrr": topk[name]["mrr"],
                "f05": float(sum(f05_data[name]) / len(f05_data[name])) if f05_data[name] else None,
                "baseline_top1_acc": majority_class_acc(name, labels),
            }
        elif t in ("continuous", "ordinal"):
            scores[name] = continuous_scores.get(name, {"type": t, "n": 0})
            scores[name]["type"] = t

    output_path = results_dir / "scores.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Scores written to {output_path}")
