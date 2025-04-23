# prompt_tools.py

import os
import json
from PyQt6.QtCore import Qt

TEMPLATE_DIR = "prompt_templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def save_prompt_template(name: str, template: str) -> None:
    """Save a single prompt template by name."""
    path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"template": template}, f, indent=2)

def load_prompt_template(name: str) -> str:
    """Load a single prompt template by name."""
    path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("template", "")
    return ""

def list_prompt_templates() -> list:
    """List all available template names."""
    return [
        f.replace(".json", "")
        for f in os.listdir(TEMPLATE_DIR)
        if f.endswith(".json")
    ]

def load_templates() -> dict:
    """
    Load all templates into a name -> template dictionary.
    Used for sidebar population.
    """
    templates = {}
    for name in list_prompt_templates():
        template = load_prompt_template(name)
        if template:
            templates[name] = template
    return templates

def apply_template(template: str, user_input: str, previous_output: str = "") -> str:
    """Apply template with input and optional previous output for chaining."""
    prompt = template.replace("{{input}}", user_input)
    prompt = prompt.replace("{{previous}}", previous_output)
    return prompt

def delete_prompt_template(name: str) -> None:
    """Delete a saved prompt template by name."""
    path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)


def chain_prompts(template_names: list[str], user_input: str, model_loader) -> str:
    """
    Run multiple templates in sequence, feeding the output to the next.
    Each step appends a prompt/response block to the final output.
    """
    if not model_loader:
        raise ValueError("ModelLoader instance is required for chaining.")

    previous_output = ""
    seen = set()
    conversation = ""

    for name in template_names:
        if name in seen:
            raise ValueError(f"Circular chaining detected with template: {name}")
        seen.add(name)

        template = load_prompt_template(name)
        if not template.strip():
            raise ValueError(f"Template '{name}' is empty or invalid.")

        prompt = apply_template(template, user_input, previous_output)
        response = model_loader.generate_single_response(prompt)
        previous_output = response

        conversation += (
            f"<div class='ai-output'>"
            f"<b style='color:#ff79c6;'>Prompt ({name}):</b><br><i>{prompt}</i><hr>"
            f"<b style='color:#8be9fd;'>Response:</b><br>{response}<hr><br></div>"
        )

    return conversation


def update_template_selector_state(self):
    any_checked = any(
        self.chain_list.item(i).checkState() == Qt.CheckState.Checked
        for i in range(self.chain_list.count())
    )
    self.template_selector.setEnabled(not any_checked)
