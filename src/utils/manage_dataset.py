# src/utils/manage_dataset.py
import os
import shutil
import logging
import pandas as pd
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    A class to manage datasets, including saving, loading, archiving, and restoring datasets.
    """

    def __init__(self, temp_dir: str, archive_dir: str, merged_dir: str = "data_bin/merged_datasets"):
        """
        Initialize the DatasetManager with temporary, archive, and merged directories.

        Args:
            temp_dir (str): Directory for temporary dataset storage.
            archive_dir (str): Directory for archived dataset storage.
            merged_dir (str): Directory for merged dataset storage.
        """
        self.temp_dir = temp_dir
        self.archive_dir = archive_dir
        self.merged_dir = merged_dir
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Ensure that the temporary, archive, and merged directories exist."""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(self.merged_dir, exist_ok=True)

    def list_datasets(self) -> List[str]:
        """
        List all datasets in the temporary directory.

        Returns:
            List[str]: List of dataset filenames.
        """
        try:
            return [
                f for f in os.listdir(self.temp_dir)
                if f.endswith((".csv", ".json"))
                and os.path.getsize(os.path.join(self.temp_dir, f)) > 0
            ]
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []

    def list_archived_datasets(self) -> List[str]:
        """
        List all archived datasets in the archive directory.

        Returns:
            List[str]: List of archived dataset filenames.
        """
        try:
            return [
                f for f in os.listdir(self.archive_dir)
                if f.endswith((".csv", ".json"))
                and os.path.getsize(os.path.join(self.archive_dir, f)) > 0
            ]
        except Exception as e:
            logger.error(f"Failed to list archived datasets: {str(e)}")
            return []

    def list_merged_datasets(self) -> List[str]:
        """
        List all merged datasets in the merged directory.

        Returns:
            List[str]: List of merged dataset filenames.
        """
        try:
            return [
                f for f in os.listdir(self.merged_dir)
                if f.endswith((".csv", ".json"))
                and os.path.getsize(os.path.join(self.merged_dir, f)) > 0
            ]
        except Exception as e:
            logger.error(f"Failed to list merged datasets: {str(e)}")
            return []

    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a dataset from the temporary directory.

        Args:
            dataset_name (str): Name of the dataset file.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if errors occur.
        """
        try:
            dataset_path = os.path.join(self.temp_dir, dataset_name)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found")
            if dataset_name.endswith(".csv"):
                return pd.read_csv(dataset_path)
            elif dataset_name.endswith(".json"):
                return pd.read_json(dataset_path)
            else:
                raise ValueError(f"Unsupported file format for dataset '{dataset_name}'")
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {str(e)}")
            return None

    def load_archived_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load an archived dataset from the archive directory.

        Args:
            dataset_name (str): Name of the archived dataset file.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if errors occur.
        """
        try:
            archived_path = os.path.join(self.archive_dir, dataset_name)
            if not os.path.exists(archived_path):
                raise FileNotFoundError(f"Archived dataset '{dataset_name}' not found")
            if dataset_name.endswith(".csv"):
                return pd.read_csv(archived_path)
            elif dataset_name.endswith(".json"):
                return pd.read_json(archived_path)
            else:
                raise ValueError(f"Unsupported file format for dataset '{dataset_name}'")
        except Exception as e:
            logger.error(f"Failed to load archived dataset '{dataset_name}': {str(e)}")
            return None

    def load_merged_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a merged dataset from the merged directory.

        Args:
            dataset_name (str): Name of the merged dataset file.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if errors occur.
        """
        try:
            merged_path = os.path.join(self.merged_dir, dataset_name)
            if not os.path.exists(merged_path):
                raise FileNotFoundError(f"Merged dataset '{dataset_name}' not found")
            if dataset_name.endswith(".csv"):
                return pd.read_csv(merged_path)
            elif dataset_name.endswith(".json"):
                return pd.read_json(merged_path)
            else:
                raise ValueError(f"Unsupported file format for dataset '{dataset_name}'")
        except Exception as e:
            logger.error(f"Failed to load merged dataset '{dataset_name}': {str(e)}")
            return None

    def save_csv_file(self, df: pd.DataFrame, filename: str):
        """
        Save a DataFrame to a CSV file in the temporary directory.

        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): Name of the file to save.
        """
        try:
            filepath = os.path.join(self.temp_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CSV file '{filename}': {str(e)}")

    def save_json_file(self, df: pd.DataFrame, filename: str):
        """
        Save a DataFrame to a JSON file in the temporary directory.

        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): Name of the file to save.
        """
        try:
            filepath = os.path.join(self.temp_dir, filename)
            df.to_json(filepath, orient="records", indent=4)
            logger.info(f"Dataset saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON file '{filename}': {str(e)}")

    def archive_dataset(self, dataset_path: str) -> bool:
        """
        Archive a dataset by moving it from the temporary directory to the archive directory.

        Args:
            dataset_path (str): Path to the dataset file to archive.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset file not found at path: {dataset_path}")
                return False

            # Extract the filename from the path
            dataset_name = os.path.basename(dataset_path)

            # Define the destination path in the archive directory
            archive_path = os.path.join(self.archive_dir, dataset_name)

            # Move the file to the archive directory
            shutil.move(dataset_path, archive_path)
            logger.info(f"Dataset '{dataset_name}' archived successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to archive dataset: {str(e)}")
            return False

    def restore_archived_dataset(self, dataset_name: str) -> bool:
        """
        Restore an archived dataset by moving it back to the temporary directory.

        Args:
            dataset_name (str): Name of the archived dataset file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            source_path = os.path.join(self.archive_dir, dataset_name)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Archived dataset '{dataset_name}' not found")
            dest_path = os.path.join(self.temp_dir, dataset_name)
            shutil.move(source_path, dest_path)
            logger.info(f"Dataset '{dataset_name}' restored successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to restore archived dataset '{dataset_name}': {str(e)}")
            return False

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset from the specified directory.

        Args:
            dataset_name (str): Name of the dataset file to delete.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(dataset_name):
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found")
            os.remove(dataset_name)
            logger.info(f"Dataset '{dataset_name}' deleted successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to delete dataset '{dataset_name}': {str(e)}")
            return False

    def merge_datasets(self, dataset_names: List[str], output_name: str) -> bool:
        """
        Merge multiple datasets into a single dataset and save it in the merged_datasets directory.

        Args:
            dataset_names (List[str]): List of dataset filenames to merge.
            output_name (str): Name of the merged dataset file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not dataset_names:
                logger.error("No datasets selected for merging.")
                return False

            dfs = []
            for dataset_name in dataset_names:
                # Determine if the dataset is archived
                is_archived = dataset_name in self.list_archived_datasets()
                is_merged = dataset_name in self.list_merged_datasets()
                dataset_path = (
                    os.path.join(self.archive_dir, dataset_name)
                    if is_archived
                    else (os.path.join(self.merged_dir, dataset_name) if is_merged else os.path.join(self.temp_dir, dataset_name))
                )

                if not os.path.exists(dataset_path):
                    logger.error(f"Dataset '{dataset_name}' not found at path: {dataset_path}")
                    return False

                try:
                    if dataset_name.endswith(".csv"):
                        df = pd.read_csv(dataset_path)
                    elif dataset_name.endswith(".json"):
                        df = pd.read_json(dataset_path)
                    else:
                        logger.error(f"Unsupported file format for dataset '{dataset_name}'")
                        return False

                    if df.empty:
                        logger.warning(f"Dataset '{dataset_name}' is empty.")
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to load dataset '{dataset_name}': {str(e)}")
                    return False

            if not dfs:
                logger.error("No valid datasets to merge.")
                return False

            # Merge datasets
            merged_df = pd.concat(dfs, ignore_index=True)

            # Save the merged dataset to the merged_datasets directory
            output_path = os.path.join(self.merged_dir, output_name)
            try:
                if output_name.endswith(".csv"):
                    merged_df.to_csv(output_path, index=False)
                elif output_name.endswith(".json"):
                    merged_df.to_json(output_path, orient="records", indent=4)
                else:
                    logger.error(f"Unsupported file format for output dataset '{output_name}'")
                    return False

                logger.info(f"Datasets merged successfully into '{output_name}'!")
                return True
            except Exception as e:
                logger.error(f"Failed to save merged dataset '{output_name}': {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Merge operation failed: {str(e)}")
            return False

    def get_temp_filename(self, prefix: str) -> str:
        """
        Generate a unique filename for temporary dataset storage.

        Args:
            prefix (str): Prefix for the filename.

        Returns:
            str: Unique filename.
        """
        import uuid
        return f"{prefix}_{uuid.uuid4().hex}.csv"