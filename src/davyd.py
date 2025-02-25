import os
import logging
import pandas as pd
from typing import List, Dict, Optional
from utils.manage_dataset import DatasetManager
from autogen_client.template_manager import TemplateManager
from autogen_client.data_validator import DataValidator
from autogen_client.visualization import Visualization
from model_providers import BaseModelClient
from autogen_client.assistant_agent import AssistantAgent
from utils.proxy_agent import ProxyAgent

# Configure logging
logger = logging.getLogger(__name__)

class Davyd:
    """
    Core class for generating and managing datasets using DAVYD.
    """

    def __init__(
        self,
        num_entries: int,
        model_client: BaseModelClient,
        model_name: str,
        dataset_manager: DatasetManager,
        section_separator: str = "|",
        data_separator: str = '"',
        max_retries: int = 2,
    ):
        """
        Initialize the Davyd dataset generator.

        Args:
            num_entries (int): Number of entries to generate.
            model_client (BaseModelClient): The model client to use for generation.
            model_name (str): Name of the model to use.
            dataset_manager (DatasetManager): DatasetManager instance for managing datasets.
            section_separator (str): The separator used to split fields (e.g., "|", ":", "-", "~").
            data_separator (str): The separator used to wrap field values (e.g., '"', "'").
            max_retries (int): Maximum number of retries if the dataset is incomplete.
        """
        self.num_entries = num_entries
        self.model_client = model_client
        self.model_name = model_name
        self.dataset_manager = dataset_manager
        self.section_separator = section_separator
        self.data_separator = data_separator
        self.max_retries = max_retries
        self.dataset: List[Dict] = []
        self.fields: List[str] = []
        self.template_manager = TemplateManager()
        self.data_validator = DataValidator()
        self.visualization = Visualization()
        logger.info(
            f"Davyd initialized with {num_entries} entries using {model_name}, "
            f"section separator '{section_separator}', and data separator '{data_separator}'"
        )

    def generate_dataset(self, heading: str, example_rows: List[str]) -> None:
        """
        Generate a dataset based on the provided structure.

        Args:
            heading (str): Separator-separated field names.
            example_rows (List[str]): Example data rows.

        Raises:
            ValueError: If the heading format is invalid.
        """
        if not self._validate_heading_format(heading):
            raise ValueError("Invalid heading format")

        self.fields = [field.strip().strip(self.data_separator) for field in heading.split(self.section_separator)]

        retries = 0
        while len(self.dataset) < self.num_entries and retries < self.max_retries:
            prompt = self._create_advanced_prompt(self.fields, example_rows)
            logger.info(f"Generating dataset with prompt: {prompt[:200]}...")

            try:
                response = self.model_client.generate_text(prompt)
                self._process_response(response, self.fields)
                if len(self.dataset) < self.num_entries:
                    retries += 1
                    logger.warning(f"Dataset incomplete. Retrying ({retries}/{self.max_retries})...")
            except Exception as e:
                logger.error(f"Dataset generation failed: {e}")
                raise

        if len(self.dataset) < self.num_entries:
            logger.warning(f"Failed to generate {self.num_entries} entries after {self.max_retries} retries.")
        else:
            logger.info(f"Generated {len(self.dataset)} entries")

    def _validate_heading_format(self, heading: str) -> bool:
        """
        Validate the format of the heading.

        Args:
            heading (str): The heading to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.section_separator in heading

    def _create_advanced_prompt(self, fields: List[str], body_examples: List[str] = None) -> str:
        """
        Create a prompt for structured data generation.

        Args:
            fields (List[str]): List of field names.
            body_examples (List[str], optional): Example rows. Defaults to None.

        Returns:
            str: Generated prompt.
        """
        prompt = f"Generate exactly {self.num_entries} {self.section_separator}-separated rows matching this structure:\n\n"
        prompt += self.section_separator.join([f'{self.data_separator}{field}{self.data_separator}' for field in fields])
        prompt += "\n\nEach row should adhere to these field descriptions:\n\n"

        # Field descriptions
        field_descriptions = {
            "text": "A meaningful sentence or statement",
            "intent": "The primary goal or purpose (e.g., 'information_request')",
            "sentiment": "Emotional tone (positive/neutral/negative)",
            "sentiment_polarity": "Numerical value between -1.0 and 1.0",
            "tone": "Delivery style (friendly/formal/urgent)",
            "category": "Domain or subject area",
            "keywords": "Key phrases summarizing the content"
        }

        for field in fields:
            description = field_descriptions.get(field.lower(), "Relevant data for this field")
            prompt += f"- {field}: {description}\n"

        if body_examples:
            prompt += "\nExample Rows:\n"
            for example in body_examples:
                prompt += example + "\n"

        prompt += "\nInstructions:\n"
        prompt += "- Maintain consistent data types\n"
        prompt += "- Ensure realistic and varied data\n"
        prompt += f"- Format: {self.section_separator}-separated values with {self.data_separator}-quoted fields\n"
        prompt += f"- Generate exactly {self.num_entries} rows\n"  # Explicit instruction

        return prompt

    def _process_response(self, response: str, fields: List[str]) -> None:
        """
        Process the model response and populate the dataset.

        Args:
            response (str): Raw response from the model.
            fields (List[str]): List of field names.
        """
        rows = response.strip().split("\n")
        for row in rows:
            try:
                entry = self._parse_example(row, fields)
                if entry and self._is_valid_entry(entry):
                    self.dataset.append(entry)
                    if len(self.dataset) >= self.num_entries:
                        break
            except Exception as e:
                logger.warning(f"Skipping invalid row: {e}")

    def _parse_example(self, example: str, fields: List[str]) -> Dict:
        """
        Parse a separator-separated row into a dictionary.

        Args:
            example (str): The row to parse.
            fields (List[str]): Field names.

        Returns:
            Dict: Parsed dictionary.

        Raises:
            ValueError: If the row has an incorrect number of values.
        """
        values = [v.strip().strip(self.data_separator) for v in example.split(self.section_separator)]
        if len(values) != len(fields):
            raise ValueError(f"Row has incorrect number of values: {example}")
        return {fields[i]: values[i] for i in range(len(fields))}

    def _is_valid_entry(self, entry: Dict) -> bool:
        """
        Validate a dataset entry.

        Args:
            entry (Dict): The entry to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        for field in self.fields:
            if field not in entry or not entry[field]:
                return False
        return True

    def get_dataset_as_df(self) -> pd.DataFrame:
        """
        Get the dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: The generated dataset.
        """
        return pd.DataFrame(self.dataset)

    def save_dataset(self, base_name: str = "generated_dataset") -> str:
        """
        Save the dataset to a file using the DatasetManager.

        Args:
            base_name (str): Base name for the file.

        Returns:
            str: Path to the saved file.
        """
        try:
            filename = self.dataset_manager.get_temp_filename(base_name)
            df = self.get_dataset_as_df()
            self.dataset_manager.save_csv_file(df, filename)
            logger.info(f"Dataset saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

    def use_autogen_for_generation(self, fields: List[str], example_rows: List[str]) -> None:
        """
        Use AutoGen to dynamically generate and validate the dataset.

        Args:
            fields (List[str]): List of field names.
            example_rows (List[str]): Example data rows.
        """
        assistant = AssistantAgent("assistant")
        user_proxy = ProxyAgent("user_proxy")

        retries = 0
        while len(self.dataset) < self.num_entries and retries < self.max_retries:
            prompt = self._create_advanced_prompt(fields, example_rows)
            logger.info(f"Using AutoGen to generate dataset with prompt: {prompt[:200]}...")

            try:
                user_proxy.initiate_chat(assistant, message=prompt)
                response = assistant.last_message()["content"]
                self._process_response(response, fields)
                if len(self.dataset) < self.num_entries:
                    retries += 1
                    logger.warning(f"Dataset incomplete. Retrying ({retries}/{self.max_retries})...")
            except Exception as e:
                logger.error(f"Dataset generation failed: {e}")
                raise

        if len(self.dataset) < self.num_entries:
            logger.warning(f"Failed to generate {self.num_entries} entries after {self.max_retries} retries.")
        else:
            logger.info(f"Generated {len(self.dataset)} entries")