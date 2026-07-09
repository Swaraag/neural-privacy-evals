from scenarios.attr_inference.utils import load_yaml, load_json, build_prompt
from config import REPO_ROOT

if __name__ == "__main__":
    config_path = REPO_ROOT / "scenarios" / "attr_inference" / "attr_inference.yaml"
    config = load_yaml(config_path)

    records = load_json(config['input']['records'])
    labels = load_json(config['input']['labels'])

    prompt_template = load_yaml(REPO_ROOT / config['prompt_template'])

    attr_lookup = {t["name"]: t for t in config["target_info"]}
    attr_names = [t["name"] for t in config["target_info"]]

    print(prompt_template["system_prompt"])

    for subj_id in records.keys() & labels.keys():
        prompt = build_prompt(prompt_template, attr_lookup, attr_names, records[subj_id], labels[subj_id])
