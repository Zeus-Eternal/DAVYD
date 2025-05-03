import json
from typing import List, Dict

class PromptEngineer:
    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict:
        """Load prompt templates from JSON files."""
        templates = {}
        template_files = [
            'intent_classification.json',
            'sentiment_analysis.json',
            'custom_template.json'
        ]
        for tf in template_files:
            try:
                with open(f'src/autogen_client/templates/{tf}') as f:
                    templates[tf.split('.')[0]] = json.load(f)
            except FileNotFoundError:
                continue
        return templates

    def build_generation_prompt(
        self,
        fields: List[str],
        examples: List[str],
        num_entries: int,
        template_name: str = 'custom_template'
    ) -> str:
        """Build enhanced generation prompt with validation instructions."""
        template = self.templates.get(template_name, {})
        base = template.get('base_prompt', "")
        prompt = base.format(
            fields=', '.join(fields),
            examples='\n'.join(examples),
            num_entries=num_entries
        )

        # Add validation rules
        validation_rules = [
            "1. Format exactly as shown in examples",
            "2. Ensure field values match their types",
            "3. Maintain realistic value distributions",
            "4. Remove any duplicate entries",
            "5. Validate all entries before returning"
        ]
        prompt += "\n\nValidation Rules:\n" + "\n".join(validation_rules)

        # Optional field descriptions
        if 'field_descriptions' in template:
            prompt += "\n\nField Specifications:\n"
            for f, desc in template['field_descriptions'].items():
                prompt += f"- {f}: {desc}\n"

        return prompt

    def build_validation_prompt(self, entry: str, fields: List[str]) -> str:
        """Create a prompt to validate a single entry."""
        return (
            f"Validate and correct this dataset entry:\n"
            f"Entry: {entry}\n\n"
            f"Required Fields: {', '.join(fields)}\n\n"
            "Check for:\n"
            "1. Proper formatting\n"
            "2. Appropriate values for each field\n"
            "3. Consistent data types\n\n"
            "Return ONLY the corrected entry or [INVALID] if unfixable."
        )
