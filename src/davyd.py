# src/davyd.py

import logging
import json
import pandas as pd
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

    def generate_dataset(self, heading: str, body_examples: list = None):
        """
        Generate a dataset using the model for real, non-repetitive data.

        :param heading: The data structure heading, in quoted, pipe-separated format.
        :param body_examples: Optional examples to guide the data generation.
        """
        if not validate_heading_format(heading):
            raise ValueError("Invalid heading format. Ensure each field is enclosed in quotes and separated by '|'.")

        # Parse and set field names dynamically
        self.fields = [field.strip().strip('"') for field in heading.split("|")]
        logging.info(f"Defined fields: {self.fields}")
        self.dataset = []

        # Create advanced prompt
        prompt = self.create_advanced_prompt(self.fields, body_examples)

        # Process provided example rows for context
        if body_examples:
            logging.info("Processing example rows for context...")
            for example in body_examples:
                try:
                    entry = self.parse_example(example, self.fields)
                    if self.is_valid_entry(entry):
                        self.dataset.append(entry)
                        logging.info(f"Added example entry: {entry}")
                    else:
                        logging.warning(f"Skipped invalid example entry: {entry}")
                except Exception as e:
                    logging.warning(f"Failed to parse example: {example} - {e}")

        # Determine how many entries need to be generated
        entries_to_generate = self.num_entries - len(self.dataset)
        logging.info(f"Generating {entries_to_generate} new entries using the model...")

        for i in range(entries_to_generate):
            try:
                # Use the advanced prompt with embedded temperature instructions
                generated_text = self.ollama_client.generate_text(prompt=prompt)
                
                # Each generated row is a pipe-separated string
                generated_rows = generated_text.strip().split("\n")
                for row in generated_rows:
                    if not row.strip():
                        continue  # Skip empty lines
                    entry = self.parse_example(row, self.fields)
                    if self.is_valid_entry(entry):
                        self.dataset.append(entry)
                        logging.info(f"Added generated entry {len(self.dataset)}: {entry}")
                        if len(self.dataset) >= self.num_entries:
                            break
                    else:
                        logging.warning(f"Skipped invalid generated entry: {entry}")
            except Exception as e:
                logging.error(f"Error generating entry {i+1}: {e}")

    def parse_example(self, example: str, fields: list) -> dict:
        """
        Parse an example row into a dictionary.

        :param example: The example row as a string.
        :param fields: List of field names.
        :return: Parsed dictionary of values.
        """
        values = [value.strip().strip('"') for value in example.split("|")]
        if len(values) != len(fields):
            logging.warning(f"Example row does not match fields: {example}")
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
            raise ValueError("Unsupported output format. Use 'csv' or 'json'.")
