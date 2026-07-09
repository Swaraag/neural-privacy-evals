import yaml
import json

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def build_prompt(prompt_template, attr_lookup, attr_names, record, label):
    user_msg = ""

    user_msg += prompt_template["header"].format(attr_labels=", ".join(attr_names))
    user_msg += "\n"

    user_msg += record
    user_msg += "\n"

    response_template = build_response_template(prompt_template["attr_entry_schema"], attr_lookup)

    attr_options = build_attr_options(attr_lookup)

    user_msg += prompt_template["footer"].format(response_template=response_template, attribute_options=attr_options)

    return (prompt_template["system_prompt"], user_msg)

def build_attr_options(attr_lookup):
    attr_options = ""
    for attr_name, attr_info in attr_lookup.items():
        if attr_info.get("choices") is not None:
            attr_options += "For " + attr_name + ", your guesses must be chosen from: " + ", ".join(attr_info["choices"]) + "."
    return attr_options

def indent(text, spaces=4):
    return "\n".join(" " * spaces + line for line in text.splitlines())

def build_response_template(attr_entry_schema, attr_lookup):
    response_template = []
    for attr_name, attr_info in attr_lookup.items():
        response_template.append(indent(attr_entry_schema.format(attr_name=attr_name, guess_structure=attr_info["guess_structure"])))
    return '{\n  "estimates": {\n' + ",\n".join(response_template) + '\n  }\n}'