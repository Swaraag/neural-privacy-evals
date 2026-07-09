from scenarios.attr_inference.utils import load_yaml, load_json, build_prompt
from config import REPO_ROOT, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from openai import OpenAI
import json

if __name__ == "__main__":
    config_path = REPO_ROOT / "scenarios" / "attr_inference" / "attr_inference.yaml"
    config = load_yaml(config_path)

    records = load_json(config['input']['records'])
    labels = load_json(config['input']['labels'])

    prompt_template = load_yaml(REPO_ROOT / config['prompt_template'])

    attr_lookup = {t["name"]: t for t in config["target_info"]}
    attr_names = [t["name"] for t in config["target_info"]]

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    subj_ids = sorted(records.keys() & labels.keys())[:config.get("max_subjects")]

    for count, subj_id in enumerate(subj_ids):
        system_prompt, user_msg = build_prompt(prompt_template, attr_lookup, attr_names, records[subj_id], labels[subj_id])

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_msg
                }])
        
        raw_response = response.choices[0].message.content
        raw_response = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
        results[subj_id] = raw_response

        if (count % 1) == 0:
            print(f"{count + 1} results recorded out of {len(subj_ids)}")

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
 