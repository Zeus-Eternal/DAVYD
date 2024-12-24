# src/utils.py

import pandas as pd
import logging
import csv
import json

def validate_heading_format(heading: str) -> bool:
    """
    Validate that the heading format is correct: each field is enclosed in quotes and separated by '|'.
    Example: "field1"|"field2"|"field3"

    :param heading: The heading string to validate.
    :return: True if valid, False otherwise.
    """
    fields = heading.split("|")
    for field in fields:
        field = field.strip()
        if not (field.startswith('"') and field.endswith('"')):
            logging.warning(f"Field '{field}' is not properly quoted.")
            return False
    return True

def convert_dict_to_df(dataset: list) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.

    :param dataset: List of dictionary entries.
    :return: pandas DataFrame.
    """
    return pd.DataFrame(dataset)

def save_csv_file(df: pd.DataFrame, filename: str):
    """
    Save a pandas DataFrame to a CSV file.

    :param df: pandas DataFrame to save.
    :param filename: Name of the CSV file.
    """
    try:
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        logging.info(f"CSV file '{filename}' saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save CSV file '{filename}': {e}")
        raise

def save_json_file(dataset: list, filename: str):
    """
    Save a list of dictionaries to a JSON file.

    :param dataset: List of dictionary entries.
    :param filename: Name of the JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        logging.info(f"JSON file '{filename}' saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save JSON file '{filename}': {e}")
        raise
