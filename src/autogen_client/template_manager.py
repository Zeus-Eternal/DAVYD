import json
import os
import logging
from typing import Dict  # Import Dict from typing

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')

class TemplateManager:
    def __init__(self):
        self.template_dir = TEMPLATE_DIR
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
        logging.info(f"Template directory set to {self.template_dir}")

    def load_templates(self) -> Dict:
        templates = {}
        for file in os.listdir(self.template_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.template_dir, file), 'r') as f:
                    templates[file.replace('.json', '')] = json.load(f)
        return templates

    def save_template(self, template_name: str, structure: Dict):
        filepath = os.path.join(self.template_dir, f"{template_name}.json")
        with open(filepath, 'w') as f:
            json.dump(structure, f, indent=4)
        logging.info(f"Template '{template_name}' saved successfully.")

    def get_template(self, template_name: str) -> Dict:
        filepath = os.path.join(self.template_dir, f"{template_name}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"Template '{template_name}' not found.")
            return {}
