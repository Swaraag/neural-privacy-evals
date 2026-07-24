from scenarios.attr_inference.utils import load_yaml, load_json, run_subject, build_prompt
from config import REPO_ROOT, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from openai import OpenAI
import hashlib
import json
import random
from universal_utils import get_next_prefix_dir, get_cur_prefix_dir
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    config_path = REPO_ROOT / "scenarios" / "attr_inference" / "attr_inference.yaml"
    config = load_yaml(config_path)

    # if you want a specific dir, change to: data_path = config["data_root"] / "version_00x/"
    data_path = get_cur_prefix_dir(REPO_ROOT / config["data_root"], "version")

    records = load_json(data_path / "records.json")

    prompt_template = load_yaml(REPO_ROOT / config['prompt_template'])

    attr_lookup = {t["name"]: t for t in config["target_info"]}
    attr_names = [t["name"] for t in config["target_info"]]

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

    output_dir = REPO_ROOT / config["output_dir"]
    run_dir = get_next_prefix_dir(output_dir, "run")
    run_dir.mkdir(parents=True, exist_ok=True)

    shots = config["experiment"].get("shots", 0)
    pool_size = config["experiment"].get("exemplar_pool_size", 50)
    seed = config["experiment"].get("exemplar_seed", 42)

    all_ids = sorted(records.keys())

    if shots > 0:
        rng = random.Random(seed)
        exemplar_pool_ids = set(rng.sample(all_ids, pool_size))
        test_ids = [sid for sid in all_ids if sid not in exemplar_pool_ids]
    else:
        exemplar_pool_ids = set()
        test_ids = all_ids

    max_subjects = config.get("max_subjects")
    subj_ids = test_ids[:max_subjects] if max_subjects is not None else test_ids

    if shots > 0:
        labels = load_json(data_path / "labels.json")
        exemplar_pool_list = sorted(exemplar_pool_ids)

        def _subj_exemplars(subj_id):
            digest = int(hashlib.md5(subj_id.encode()).hexdigest(), 16) % (2 ** 31)
            rng_s = random.Random(seed ^ digest)
            chosen = rng_s.sample(exemplar_pool_list, shots)
            return [(records[cid], labels[cid]) for cid in chosen]

        subject_exemplars = {sid: _subj_exemplars(sid) for sid in subj_ids}
        with open(run_dir / "exemplar_pool.json", "w", encoding="utf-8") as f:
            json.dump(sorted(exemplar_pool_ids), f, indent=2)
    else:
        subject_exemplars = {}

    meta = {
        "model": config["model"],
        "config_version": config.get("config_version"),
        "experiment": config.get("experiment", {}),
        "data_dir": str(data_path),
        "config": config
    }

    results = {}
    failed = {}

    #print(build_prompt(prompt_template, attr_lookup, attr_names, records[subj_ids[0]], config)[1])

    # dump one realized sample prompt so the exact model-facing input under
    # this config is recoverable later. Uses first subject id
    sample_id = subj_ids[0]
    _, sample_user_prompt = build_prompt(
        prompt_template, attr_lookup, attr_names, records[sample_id], config,
        subject_exemplars.get(sample_id)
    )
    with open(run_dir / "sample_prompt.txt", "w", encoding="utf-8") as f:
        f.write(f"# sample prompt for subject {sample_id}\n\n{sample_user_prompt}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(
                run_subject,
                subj_id, records[subj_id], client, config, prompt_template, attr_lookup, attr_names,
                subject_exemplars.get(subj_id)
            ): subj_id
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