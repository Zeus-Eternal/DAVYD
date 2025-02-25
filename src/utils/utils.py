import os
import shutil
from datetime import datetime, timedelta
import logging
import pandas as pd
import json
from typing import List, Dict, Any
import streamlit as st
from utils.manage_dataset import DatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_BIN_DIR = "data_bin"
TEMP_DIR = os.path.join(DATA_BIN_DIR, "temp")
ARCHIVE_DIR = os.path.join(DATA_BIN_DIR, "archive")

# Path to save settings and datasets
SETTINGS_FILE = "settings.json"
DATASETS_DIR = TEMP_DIR
ARCHIVED_DATASETS_DIR = ARCHIVE_DIR

# Ensure directories exist
os.makedirs(DATA_BIN_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def get_temp_filename(base_name: str) -> str:
    """Generate a temporary filename with a timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
        filename = f"{timestamp}_{safe_name}"
        full_path = os.path.join(TEMP_DIR, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path
    except Exception as e:
        logger.error(f"Filename generation failed: {str(e)}")
        raise

def archive_temp_files(max_age_hours: int = 12):
    """Archive temporary files older than `max_age_hours`."""
    current_time = datetime.now()
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if current_time - file_time > timedelta(hours=max_age_hours):
            archive_path = os.path.join(ARCHIVE_DIR, filename)
            shutil.move(file_path, archive_path)
            logger.info(f"Archived {filename} to {archive_path}")

def archive_dataset(dataset_path: str):
    """Archive a dataset by moving it to the archive directory."""
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File {dataset_path} not found")
        filename = os.path.basename(dataset_path)
        dest_path = os.path.join(ARCHIVE_DIR, filename)
        shutil.move(dataset_path, dest_path)
        logger.info(f"Dataset '{filename}' archived successfully!")
        st.success(f"Dataset '{filename}' archived successfully!")
    except Exception as e:
        logger.error(f"Failed to archive dataset '{dataset_path}': {e}")
        st.error(f"Failed to archive dataset: {str(e)}")

def save_csv_file(df: pd.DataFrame, filename: str):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(filename, index=False)
        logger.info(f"CSV file saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV file {filename}: {e}")
        raise

def save_json_file(data: Dict, filename: str):
    """Save a dictionary to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {filename}: {e}")
        raise

def load_csv_file(filename: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with comprehensive error handling."""
    try:
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return pd.DataFrame()
        if os.path.getsize(filename) == 0:
            logger.error(f"Empty file: {filename}")
            return pd.DataFrame()
        df = pd.read_csv(filename)
        if df.empty:
            logger.warning(f"Empty dataset: {filename}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Malformed CSV: {filename}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Load failed for {filename}: {str(e)}")
        return pd.DataFrame()

def load_json_file(filename: str) -> dict:
    """Load a JSON file into a dictionary."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        logger.info(f"JSON file {filename} loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {filename}: {e}")
        raise

def validate_heading_format(heading: str, section_separator: str = "|") -> bool:
    """Validate the format of the heading string."""
    try:
        if not heading or not isinstance(heading, str):
            return False
        fields = heading.split(section_separator)
        if len(fields) < 2:
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to validate heading format: {e}")
        return False

def parse_example(example: str, fields: List[str], section_separator: str = "|", data_separator: str = '"') -> Dict:
    """Parse a separator-separated row into a dictionary."""
    try:
        example = example.strip()
        if example.startswith(data_separator) and example.endswith(data_separator):
            example = example[1:-1]
        values = [value.strip().strip(data_separator) for value in example.split(section_separator)]
        if len(values) != len(fields):
            raise ValueError(f"Row has {len(values)} values but expected {len(fields)}")
        return {field: value for field, value in zip(fields, values)}
    except Exception as e:
        logger.error(f"Failed to parse example: {e}")
        raise

def convert_dict_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert a dictionary to a DataFrame."""
    try:
        df = pd.DataFrame(data)
        logger.info("Dictionary converted to DataFrame successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to convert dictionary to DataFrame: {e}")
        raise

def download_dataset(df: pd.DataFrame, filename: str, format: str):
    """Download dataset in the specified format."""
    try:
        temp_filename = get_temp_filename(filename)
        if format == "csv":
            df.to_csv(temp_filename, index=False)
        elif format == "json":
            df.to_json(temp_filename, orient="records", indent=4)
        elif format == "xlsx":
            df.to_excel(temp_filename, index=False)
        elif format == "txt":
            df.to_csv(temp_filename, index=False, sep="\t")
        with open(temp_filename, 'rb') as file:
            st.download_button(
                label=f"📥 Download as {format.upper()}",
                data=file,
                file_name=filename,
                mime='application/octet-stream'
            )
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        st.error(f"Failed to download dataset: {e}")

def merge_datasets(datasets_to_merge: List[str], merged_name: str):
    """Merge multiple datasets into a single dataset."""
    try:
        dataframes = []
        for dataset_name in datasets_to_merge:
            dataset_path = os.path.join(DATASETS_DIR, dataset_name)
            try:
                df = pd.read_csv(dataset_path)
                if df.empty:
                    raise pd.errors.EmptyDataError(f"DataFrame from {dataset_name} is empty.")
                if len(df.columns) == 0:
                    raise ValueError(f"DataFrame from {dataset_name} has no columns.")
                dataframes.append(df)
                logger.info(f"Loaded {dataset_name}, shape: {df.shape}")
            except pd.errors.EmptyDataError as e:
                logger.error(f"Error loading {dataset_name}: {e}")
                st.error(f"Error loading {dataset_name}: {e}")
                return
            except ValueError as e:
                logger.error(f"Error loading {dataset_name}: {e}")
                st.error(f"Error loading {dataset_name}: {e}")
                return
            except pd.errors.ParserError as e:
                logger.error(f"Error parsing {dataset_name}: {e}")
                st.error(f"Error parsing {dataset_name}: {e}")
                return
            except FileNotFoundError as e:
                logger.error(f"File not found {dataset_name}: {e}")
                st.error(f"File not found {dataset_name}: {e}")
                return
            except Exception as e:
                logger.error(f"Unexpected error loading {dataset_name}: {e}")
                st.error(f"Unexpected error loading {dataset_name}: {e}")
                return

        if not dataframes:
            raise ValueError("No valid dataframes to merge.")

        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_path = os.path.join(DATASETS_DIR, merged_name)
        merged_df.to_csv(merged_path, index=False)
        logger.info(f"Merged datasets saved to: {merged_path}")
        st.success(f"Datasets merged successfully as '{merged_name}'!")
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        st.error(f"Failed to merge datasets: {e}")

def manage_datasets(dataset_manager: DatasetManager):
    """Manage datasets (view, edit, delete, archive, merge)."""
    st.header("📂 Dataset Management")

    try:
        datasets = [f for f in os.listdir(dataset_manager.temp_dir) if f.endswith((".csv", ".json")) and os.path.getsize(os.path.join(dataset_manager.temp_dir, f)) > 0]
    except Exception as e:
        st.error(f"Failed to list datasets: {str(e)}")
        return

    if not datasets:
        st.warning("No valid datasets available")
        return

    selected_dataset = st.selectbox("Select Dataset", datasets)
    dataset_path = os.path.join(dataset_manager.temp_dir, selected_dataset)

    # Load dataset with error handling
    try:
        if selected_dataset.endswith(".csv"):
            df = load_csv_file(dataset_path)
        elif selected_dataset.endswith(".json"):
            df = load_json_file(dataset_path)
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        df = pd.DataFrame()

    # Display validation
    if df.empty:
        st.warning("This dataset contains no valid data")
        if st.button("🗑️ Delete Invalid Dataset"):
            try:
                os.remove(dataset_path)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Deletion failed: {str(e)}")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(50))

        # Data Editor
        st.subheader("Edit Dataset")
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        )

        # Save edited dataset
        if st.button("💾 Save Edits"):
            try:
                if selected_dataset.endswith(".csv"):
                    edited_df.to_csv(dataset_path, index=False)
                elif selected_dataset.endswith(".json"):
                    edited_df.to_json(dataset_path, orient="records", indent=4)
                st.success("Dataset saved successfully!")
            except Exception as e:
                st.error(f"Failed to save dataset: {str(e)}")

    # Archive dataset
    if st.button("📦 Archive Dataset", key="archive_dataset_management"):
        try:
            dataset_manager.archive_dataset(dataset_path)
            st.success(f"Dataset '{selected_dataset}' archived successfully!")
        except Exception as e:
            st.error(f"Error archiving dataset: {str(e)}")

    # Merge datasets
    st.subheader("Merge Datasets")
    datasets_to_merge = st.multiselect("Select Datasets to Merge", datasets)
    merged_name = st.text_input("Merged Dataset Name", "merged_dataset.csv")
    if st.button("🔀 Merge Datasets"):
        if not datasets_to_merge:
            st.warning("Please select at least one dataset to merge.")
        else:
            try:
                merge_datasets(datasets_to_merge, merged_name)
                st.success(f"Datasets merged successfully as '{merged_name}'!")
            except Exception as e:
                st.error(f"Error merging datasets: {str(e)}")



                