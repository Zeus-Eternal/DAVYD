# src/davyd.py

import logging
import random
import json
import pandas as pd
from typing import List, Dict
from ollama_client import OllamaClient
from utils import (
    validate_heading_format,
    convert_dict_to_df,
    save_csv_file,
    save_json_file
)

class Davyd:
    def __init__(self, num_entries: int, ollama_host: str, ollama_model: str):
        """
        Initialize the Davyd class for real data generation.

        :param num_entries: Number of entries to generate.
        :param ollama_host: The API URL for the Ollama service.
        :param ollama_model: The Ollama model to use.
        """
        self.num_entries = num_entries
        self.ollama_client = OllamaClient(model=ollama_model, host=ollama_host)
        self.dataset = []
        self.fields = []
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Davyd initialized with {self.num_entries} entries, model '{ollama_model}', and host '{ollama_host}'.")

    def create_advanced_prompt(self, fields: list, body_examples: list = None) -> str:
        """
        Create an advanced prompt template for structured pipe-separated data generation.

        :param fields: List of field names.
        :param body_examples: Optional example entries.
        :return: Generated prompt string.
        """
        prompt = "Generate a list of pipe-separated rows matching the following structure:\n\n"
        prompt += "|".join([f'"{field}"' for field in fields])
        prompt += "\n\nEach row should adhere to the field descriptions below:\n\n"

        prompt += "### Field Descriptions:\n"
        # Example field descriptions
        for field in fields:
            if field.lower() == "text":
                description = "A meaningful sentence or statement relevant to the context."
            elif field.lower() == "intent":
                description = "The primary goal or purpose of the text, such as 'information_request' or 'celebration'."
            elif field.lower() == "sentiment":
                description = "The overall emotional tone, e.g., 'positive', 'neutral', or 'negative'."
            elif field.lower() == "sentiment_polarity":
                description = "A numerical value between -1.0 (strongly negative) and 1.0 (strongly positive)."
            elif field.lower() == "tone":
                description = "The way the message is delivered, such as 'friendly', 'inquisitive', or 'urgent'."
            elif field.lower() == "category":
                description = "The domain or subject area, e.g., 'technology', 'food', or 'personal'."
            elif field.lower() == "keywords":
                description = "Key phrases or terms that summarize the entry's topic."
            else:
                description = "Provide an appropriate description."
            prompt += f"- **{field}**: {description}\n"

        prompt += "\n"

        if body_examples:
            prompt += "### Example Rows:\n"
            for example in body_examples:
                try:
                    # Assume example is already pipe-separated string
                    prompt += example + "\n"
                except Exception as e:
                    logging.warning(f"Failed to parse example: {example} - {e}")

        # Embed temperature instruction in the prompt
        prompt += "\n### Instructions:\n"
        prompt += "Ensure the generated data has a creativity level similar to a temperature setting of 0.5.\n"
        prompt += "Please generate the required number of pipe-separated rows following the structure and descriptions above."

        return prompt

    def generate_dataset(self, heading: str, example_rows: List[str]):
        """
        Generate a dataset based on the provided structure and examples.

        :param heading: A pipe-separated string of field names.
        :param example_rows: A list of example data rows.
        """
        fields = [field.strip().strip('"') for field in heading.split('|')]
        self.fields = fields  # Store fields for validation
        logging.info(f"Generating dataset with fields: {fields}")

        prompt = self.create_advanced_prompt(fields, example_rows)
        response = self.ollama_client.generate_text(prompt)

        if not response:
            logging.error("No response from Ollama. Dataset generation aborted.")
            return

        rows = response.strip().split("\n")
        for row in rows:
            entry = self.parse_example(row, fields)
            if entry and self.is_valid_entry(entry):
                self.dataset.append(entry)

        logging.info(f"Dataset generation completed. Total entries: {len(self.dataset)}")

    def parse_example(self, example: str, fields: list) -> dict:
        """
        Parse an example row into a dictionary.

        :param example: The example row as a string.
        :param fields: List of field names.
        :return: Parsed dictionary of values.
        """
        values = [value.strip().strip('"') for value in example.split("|")]
        if len(values) != len(fields):
            logging.warning(f"Row does not match fields: {example}")
            return {}
        return {field: value for field, value in zip(fields, values)}

    def is_valid_entry(self, entry: dict) -> bool:
        """
        Validate the completeness and consistency of an entry.

        :param entry: Dictionary entry to validate.
        :return: True if entry is valid, False otherwise.
        """
        required_fields = self.fields
        for field in required_fields:
            if field not in entry or not entry[field]:
                logging.warning(f"Missing or empty field '{field}' in entry: {entry}")
                return False

        # Validate sentiment_polarity if present
        if "sentiment_polarity" in entry:
            try:
                sentiment_polarity = float(entry.get("sentiment_polarity", 0))
                sentiment = entry.get("sentiment", "").lower()
                if sentiment == "positive" and not (0.5 <= sentiment_polarity <= 1.0):
                    logging.warning(f"Inconsistent sentiment_polarity '{sentiment_polarity}' for positive sentiment: {entry}")
                    return False
                elif sentiment == "negative" and not (-1.0 <= sentiment_polarity <= -0.5):
                    logging.warning(f"Inconsistent sentiment_polarity '{sentiment_polarity}' for negative sentiment: {entry}")
                    return False
                elif sentiment == "neutral" and not (-0.5 < sentiment_polarity < 0.5):
                    logging.warning(f"Inconsistent sentiment_polarity '{sentiment_polarity}' for neutral sentiment: {entry}")
                    return False
            except ValueError:
                logging.warning(f"Non-numeric sentiment_polarity '{entry.get('sentiment_polarity')}' in entry: {entry}")
                return False

        # Additional validation can be added here (e.g., keyword formatting)
        return True

    def get_dataset_as_df(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        :return: DataFrame of the dataset.
        """
        return convert_dict_to_df(self.dataset)

    def save_dataset(self, filename: str = "generated_dataset.csv", output_format: str = "csv"):
        """
        Save the dataset in the specified format.

        :param filename: File name for saving.
        :param output_format: Format ('csv' or 'json').
        """
        if output_format == "csv":
            df = self.get_dataset_as_df()
            save_csv_file(df, filename)
            logging.info(f"Dataset saved as {filename} in CSV format.")
        elif output_format == "json":
            save_json_file(self.dataset, filename)
            logging.info(f"Dataset saved as {filename} in JSON format.")
        else:
            logging.error("Unsupported output format. Use 'csv' or 'json'.")
            raise ValueError("Unsupported output format. Use 'csv' or 'json'.")

# Note: Ensure that the above class definition is the only one in davyd.py
