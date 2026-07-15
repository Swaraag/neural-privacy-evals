from scenarios.attr_inference.utils import load_yaml, load_json, run_subject, build_prompt
from config import REPO_ROOT, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from openai import OpenAI
import json
from universal_utils import get_next_run_dir
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    config_path = REPO_ROOT / "scenarios" / "attr_inference" / "attr_inference.yaml"
    config = load_yaml(config_path)

    records = load_json(config['input']['records'])

    prompt_template = load_yaml(REPO_ROOT / config['prompt_template'])

    attr_lookup = {t["name"]: t for t in config["target_info"]}
    attr_names = [t["name"] for t in config["target_info"]]

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

    output_dir = REPO_ROOT / config["output_dir"]
    run_dir = get_next_run_dir(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    subj_ids = sorted(records.keys())[:config.get("max_subjects")] if config.get("max_subjects") is not None else sorted(records.keys())

    meta = {
        "model": config["model"],
        "prompt_template": str(config["prompt_template"]),
        "system_prompt": prompt_template["system_prompt"],
        "header": prompt_template["header"],
        "footer": prompt_template["footer"],
        "max_subjects": config.get("max_subjects"),
    }

    results = {}
    failed = {}

    # print(build_prompt(prompt_template, attr_lookup, attr_names, records[subj_id[0]])[1])

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(run_subject, subj_id, records[subj_id], client, config, prompt_template, attr_lookup, attr_names): subj_id
            for subj_id in subj_ids
        }
        for count, future in enumerate(as_completed(futures)):
            subj_id, result, error = future.result()
            if error:
                failed[subj_id] = error
                print(f"{count + 1}/{len(subj_ids)} FAILED: {subj_id} — {error}")
                with open(run_dir / "failed.json", "w") as f:
                    json.dump(failed, f, indent=2)
            else:
                results[subj_id] = result
                print(f"{count + 1}/{len(subj_ids)} complete")
            if (count + 1) % 20 == 0 or (count + 1) == len(subj_ids):
                elapsed = time.time() - start_time
                print(f"--- {count + 1}/{len(subj_ids)} done | {elapsed:.1f}s elapsed ---")
    print(f"Run complete in {time.time() - start_time:.1f}s")
    # saving results and associated config file
    with open(run_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump({"meta": meta, **results}, file, indent=2)
    shutil.copy(config_path, run_dir / "config.yaml")
    
    if failed:
        with open(run_dir / "failed.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(f"{len(failed)} subjects failed — see failed.json")