import yaml
import json

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

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


def run_subject(subj_id, record, client, config, prompt_template, attr_lookup, attr_names):
    try:
        system_prompt, user_msg = build_prompt(prompt_template, attr_lookup, attr_names, record, config)
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
    """TO DO"""
    return record

def flatten(record):
    """TO DO"""
    return record

def build_prompt(prompt_template, attr_lookup, attr_names, record, config):
    """Called by run_subject()"""
    user_msg = ""

    user_msg += prompt_template["header"].format(attr_labels=", ".join(attr_names))
    user_msg += "\n"

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

def process_categorical(attr_name, attr_inference, label, topk, mrr_data):
    guess = attr_inference.get('guesses') or attr_inference.get('guess')
    if not isinstance(guess, list):
        guess = [guess]

    type_fn = type(label)
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