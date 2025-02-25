# src/autogen/field_suggester.py

import logging
from typing import List, Dict
from ollama_client import OllamaClient

class FieldSuggester:
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    def suggest_fields(self, user_input: str) -> List[Dict]:
        prompt = f"Based on the following description, suggest field names and their data types:\n\n{user_input}"
        try:
            response = self.client.generate_text(prompt)
            suggestions = response.strip().split('\n')
            fields = []
            for suggestion in suggestions:
                name, dtype = suggestion.split(':')
                fields.append({'name': name.strip(), 'type': dtype.strip()})
            return fields
        except Exception as e:
            logging.error(f"Error in suggesting fields: {e}")
            return []

    def suggest_data_types(self, fields: List[str]) -> List[str]:
        # Implement logic to suggest data types based on field names
        # This can be expanded with more sophisticated AI-based suggestions
        type_mapping = {
            'text': 'string',
            'intent': 'string',
            'sentiment': 'string',
            'sentiment_polarity': 'float',
            'tone': 'string',
            'category': 'string',
            'keywords': 'list'
        }
        return [type_mapping.get(field.lower(), 'string') for field in fields]
