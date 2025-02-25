# src/autogen/data_validator.py

import logging
from typing import List, Dict

class DataValidator:
    def validate_entry(self, entry: Dict, fields: List[Dict]) -> bool:
        try:
            for field in fields:
                name = field['name']
                dtype = field['type']
                if name not in entry:
                    logging.warning(f"Missing field '{name}' in entry.")
                    return False
                value = entry[name]
                if dtype == 'float':
                    float(value)  # Will raise ValueError if not convertible
                elif dtype == 'int':
                    int(value)
                elif dtype == 'string':
                    if not isinstance(value, str):
                        raise ValueError(f"Field '{name}' should be a string.")
                elif dtype == 'list':
                    if not isinstance(value, list):
                        raise ValueError(f"Field '{name}' should be a list.")
            return True
        except Exception as e:
            logging.warning(f"Validation error: {e}")
            return False

    def validate_dataset(self, dataset: List[Dict], fields: List[Dict]) -> List[str]:
        errors = []
        for idx, entry in enumerate(dataset, start=1):
            if not self.validate_entry(entry, fields):
                errors.append(f"Entry {idx} failed validation.")
        return errors
