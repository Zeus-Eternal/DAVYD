#!/usr/bin/env python3
# src/dataset_generation.py

import logging
import json
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox

import pandas as pd
from davyd import Davyd
from model_providers import get_model_client
from utils.manage_dataset import DatasetManager

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path("settings.json")


class DatasetGenerationThread(QThread):
    """
    Robust dataset generation thread with proper error handling and configuration.
    """
    progress_updated = Signal(int, str)
    generation_complete = Signal(pd.DataFrame)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = self._load_settings()
        self._validate_settings()
        self._init_components()

    def _load_settings(self) -> Dict[str, Any]:
        """Load and validate settings with proper error handling"""
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)

            required = ["model_provider", "model_name", "fields"]
            for field in required:
                if field not in settings:
                    raise ValueError(f"Missing required setting: {field}")

            # Defaults
            settings.setdefault("section_separator", "|")
            settings.setdefault("data_separator", '"')
            settings.setdefault("quality_level", 2)
            settings.setdefault("num_entries", 10)

            return settings

        except Exception as e:
            logger.error("Failed to load settings", exc_info=True)
            raise RuntimeError(f"Settings error: {str(e)}")

    def _validate_settings(self):
        """Validate settings before initialization"""
        if not isinstance(self.settings["fields"], list) or not self.settings["fields"]:
            raise ValueError("Fields must be a non-empty list")

        if self.settings["num_entries"] <= 0:
            raise ValueError("Number of entries must be positive")

    def _init_components(self):
        """Initialize model client and dataset manager with error handling"""
        try:
            self.settings["model_client"] = get_model_client(
                self.settings["model_provider"],
                api_key=self.settings.get("api_key", "")
            )

            self.settings["dataset_manager"] = DatasetManager(
                temp_dir="data_bin/temp",
                archive_dir="data_bin/archive",
                merged_dir="data_bin/merged_datasets",
            )

            if not self.settings["model_client"].health_check():
                raise ConnectionError("Model provider is not available")

        except Exception as e:
            logger.error("Component initialization failed", exc_info=True)
            raise RuntimeError(f"Initialization error: {str(e)}")

    def run(self):
        """Main execution with comprehensive error handling"""
        try:
            dav = Davyd(
                num_entries=self.settings["num_entries"],
                model_client=self.settings["model_client"],
                model_name=self.settings["model_name"],
                dataset_manager=self.settings["dataset_manager"],
                section_separator=self.settings["section_separator"],
                data_separator=self.settings["data_separator"],
                quality_level=self.settings["quality_level"]
            )

            sep = self.settings["section_separator"]
            wrapper = self.settings["data_separator"]
            heading = sep.join(f"{wrapper}{f}{wrapper}" for f in self.settings["fields"])

            df = dav.generate_dataset(
                heading=heading,
                example_rows=self.settings.get("examples", [])
            )

            self.generation_complete.emit(df)

        except Exception as e:
            logger.error("Dataset generation failed", exc_info=True)
            self.error_occurred.emit(f"Generation error: {str(e)}")
