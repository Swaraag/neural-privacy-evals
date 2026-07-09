import yaml
import json

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def build_prompt(prompt_template, attr_lookup, attr_names, record, label):
    prompt = ""

    prompt += prompt_template["system_prompt"] 
    prompt += "\n"


    
    prompt += prompt_template["header"].format(attr_labels=", ".join(attr_names))
    prompt += "\n"

    prompt += record
    prompt += "\n"

    response_template = build_response_template(prompt_template["attr_entry_schema"], attr_lookup)

    prompt += prompt_template["footer"].format(response_template=response_template)

    return prompt

def indent(text, spaces=4):
    return "\n".join(" " * spaces + line for line in text.splitlines())

def build_response_template(attr_entry_schema, attr_lookup):
    response_template = []
    for attr_name, attr_info in attr_lookup.items():
        response_template.append(indent(attr_entry_schema.format(attr_name=attr_name, guess_structure=attr_info["guess_structure"])))
    return '{\n  "estimates": {\n' + ",\n".join(response_template) + '\n  }\n}'