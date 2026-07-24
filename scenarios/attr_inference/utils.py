import yaml
import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

CHANNEL_ORDER = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
 
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

CONDITION_LABELS = {
    "EC": "Eyes Closed",
    "EO": "Eyes Open",
}

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5)
)
def call_api(client, config, system_prompt, user_msg):
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        timeout=60
    )
    raw = response.choices[0].message.content or ""
    return raw.strip().removeprefix("```json").removesuffix("```").strip()


def run_subject(subj_id, record, client, config, prompt_template, attr_lookup, attr_names, exemplars=None):
    try:
        system_prompt, user_msg = build_prompt(prompt_template, attr_lookup, attr_names, record, config, exemplars)
        raw = call_api(client, config, system_prompt, user_msg)
        return subj_id, json.loads(raw), None
    except Exception as e:
        return subj_id, None, str(e)

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def format_spectral_table(record):
    sections = []
 
    for condition in ["EC", "EO"]:
        condition_data = record.get(condition, {})
        header = f"**{CONDITION_LABELS[condition]} Spectral Features (\u03bcV\u00b2/Hz):**"
 
        col_headers = ["Channel"] + BANDS + ["Total"]
        separator = ["---"] + ["---:"] * (len(BANDS) + 1)  # right-align numeric cols

        rows = []
        for channel in CHANNEL_ORDER:
            if channel not in condition_data:
                continue
            band_data = condition_data[channel]
            values = [band_data.get(band, 0.0) for band in BANDS]
            total = round(sum(values), 2)
            row = [channel] + [str(v) for v in values] + [str(total)]
            rows.append(row)
 
        def pipe_row(cells):
            return "| " + " | ".join(cells) + " |"
 
        table_lines = (
            [header, ""]
            + [pipe_row(col_headers)]
            + [pipe_row(separator)]
            + [pipe_row(row) for row in rows]
        )
 
        sections.append("\n".join(table_lines))
 
    return "\n\n".join(sections)

def flatten(record):
    lines = []
 
    for condition in ["EC", "EO"]:
        condition_data = record.get(condition, {})
        lines.append(f"{CONDITION_LABELS[condition]} Spectral Features (\u03bcV\u00b2/Hz):")
 
        for channel in CHANNEL_ORDER:
            if channel not in condition_data:
                continue
            band_data = condition_data[channel]
            band_str = ", ".join(
                f"{band}={band_data[band]}"
                for band in BANDS
                if band in band_data
            )
            lines.append(f"Channel {channel}: {band_str}")
 
        lines.append("")  # blank line between conditions
 
    return "\n".join(lines).strip()

def format_label_value(attr_name, val):
    if attr_name == "gender":
        label = "Male" if int(val) == 1 else "Female"
        return f"{int(val)} ({label})"
    if attr_name in ("age", "education"):
        return str(int(round(float(val))))
    return str(val)

def build_exemplar_block(exemplars, attr_names, config):
    k = len(exemplars)
    parts = []
    for i, (record, labels) in enumerate(exemplars):
        lines = [f"--- Exemplar {i + 1} of {k} ---", ""]
        if config["experiment"]["formatting"] == "flat":
            lines.append(flatten(record))
        elif config["experiment"]["formatting"] == "markdown":
            lines.append(format_spectral_table(record))
        lines += ["", "Labels:"]
        for attr_name in attr_names:
            val = labels.get(attr_name)
            if val is not None:
                lines.append(f"  {attr_name}: {format_label_value(attr_name, val)}")
        lines.append("")
        parts.append("\n".join(lines))
    return "\n".join(parts)

def build_prompt(prompt_template, attr_lookup, attr_names, record, config, exemplars=None):
    """Called by run_subject()"""
    user_msg = ""
    shots = len(exemplars) if exemplars else 0

    user_msg += prompt_template["header"].format(attr_labels=", ".join(attr_names))
    user_msg += "\n"

    if shots > 0:
        user_msg += (
            f"\nYou will first see {shots} labeled example{'s' if shots != 1 else ''} "
            f"demonstrating the task. Study each example and its ground-truth labels carefully, "
            f"then estimate attributes for the target subject at the end.\n\n"
        )
        user_msg += build_exemplar_block(exemplars, attr_names, config)
        user_msg += "--- Target subject (estimate attributes below) ---\n\n"

    user_msg += flatten(record) if config["experiment"]["formatting"] == "flat" else format_spectral_table(record) if config["experiment"]["formatting"] == "markdown" else ""
    user_msg += "\n"

    response_template = build_response_template(prompt_template["attr_entry_schema"], attr_lookup)
    attr_options = build_attr_options(attr_lookup)
    user_msg += prompt_template["footer"].format(response_template=response_template, attribute_options=attr_options)

    return (prompt_template["system_prompt"], user_msg)

def build_attr_options(attr_lookup):
    attr_options = ""
    for attr_name, attr_info in attr_lookup.items():
        if attr_info.get("choices") is not None:
            attr_options += "For " + attr_name + ", your guesses must be chosen from: " + ", ".join(attr_info["choices"]) + ".\n"
    return attr_options

def indent(text, spaces=4):
    return "\n".join(" " * spaces + line for line in text.splitlines())

def build_response_template(attr_entry_schema, attr_lookup):
    response_template = []
    for attr_name, attr_info in attr_lookup.items():
        response_template.append(indent(attr_entry_schema.format(attr_name=attr_name, guess_structure=attr_info["guess_structure"])))
    return '{\n  "estimates": {\n' + ",\n".join(response_template) + '\n  }\n}'

def process_categorical(attr_name, attr_inference, label, topk, mrr_data, f05_data, target_info, subj_id):
    guess = attr_inference.get('guesses') or attr_inference.get('guess')
    if not isinstance(guess, list):
        guess = [guess]

    type_fn = type(label)

    if attr_name in target_info and "choices" in target_info[attr_name]:
        valid = set(target_info[attr_name]["choices"])
        invalid = [g for g in guess if g not in valid]
        if invalid:
            print(f"[INVALID GUESS] {subj_id} | {attr_name}: {invalid} (valid: {valid})")

    try:
        guess = [type_fn(g) for g in guess]
    except (ValueError, TypeError):
        guess = []
    
    topk[attr_name]['total'] += 1
    
    if len(guess) > 0 and guess[0] == label:
        topk[attr_name]['correct-1'] += 1
    if len(guess) > 1 and label in guess[:2]:
        topk[attr_name]['correct-2'] += 1
    if len(guess) > 2 and label in guess[:3]:
        topk[attr_name]['correct-3'] += 1
    
    # MRR: find rank of correct label in guess list
    rank = next((i + 1 for i, g in enumerate(guess) if g == label), None)
    mrr_data[attr_name].append(float(1 / rank if rank is not None else 0))

    # F_0.5 (precision-weighted, penalizes hedging)
    hit = label in guess
    if hit and len(guess) > 0:
        precision = 1.0 / len(guess)
        # recall = 1.0 (single relevant label, and it was found)
        f05 = 1.25 * precision / (0.25 * precision + 1.0)
    else:
        f05 = 0.0
    f05_data[attr_name].append(f05)

def process_continuous(attr_name, attr_inference, label, continuous_data):
    guess = attr_inference.get('guess')
    try:
        predicted = float(guess)
        actual = float(label)
        continuous_data[attr_name].append((predicted, actual))
    except (TypeError, ValueError) as e:
        # unparseable guesses
        print(f"Unparseable guess with attribute {attr_name}. Error: {e}")

def majority_class_acc(attr_name, labels):
    vals = [v[attr_name] for v in labels.values() if v.get(attr_name) is not None]
    most_common = max(set(vals), key=vals.count)
    return float(vals.count(most_common) / len(vals))