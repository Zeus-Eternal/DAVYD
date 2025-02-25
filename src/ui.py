# src/ui.py
import os
import json
import logging
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict
from davyd import Davyd
from utils.manage_dataset import DatasetManager
from model_providers import (
    OllamaClient, DeepSeekClient, GeminiClient, ChatGPTClient,
    AnthropicClient, ClaudeClient, MistralClient, GroqClient, HuggingFaceClient
)
from autogen_client.visualization import Visualization

# Constants
TEMP_DIR = "data_bin/temp"
ARCHIVE_DIR = "data_bin/archive"
MERGED_DIR = "data_bin/merged_datasets"
SETTINGS_FILE = "settings.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetUI:
    """Central class handling all UI components and business logic"""

    def __init__(self):
        self.dataset_manager = DatasetManager(TEMP_DIR, ARCHIVE_DIR, MERGED_DIR)
        self.visualization = Visualization()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        defaults = {
            'fields': ["text", "intent", "sentiment", "sentiment_polarity", "tone", "category", "keywords"],
            'examples': [
                '"Hi there!"', '"greeting"', '"positive"', '0.9',
                '"friendly"', '"interaction"', '"hi" "hello" "welcome"'
            ],
            'data_separator': '"',
            'section_separator': "|",
            'dataset': [],
            'temp_filename': None,
            'selected_model': None,
            'is_generated': False,  # Track if dataset is generated
            'merged_datasets': [],  # Track datasets
            'selected_dataset': None  # Track the selected dataset
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def load_settings(self, provider: str) -> dict:
        """Load settings for a specific provider"""
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f).get(provider, {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception as e:
            logger.error(f"Settings load error: {e}")
            return {}

    def save_settings(self, provider: str, settings: dict):
        """Save provider-specific settings"""
        try:
            all_settings = self.load_settings("__all__") or {}
            all_settings[provider] = settings
            with open(SETTINGS_FILE, "w") as f:
                json.dump(all_settings, f, indent=4)
            st.success(f"{provider} settings saved!")
        except Exception as e:
            logger.error(f"Settings save error: {e}")
            st.error("Failed to save settings")

    def model_config_sidebar(self):
        """Render model configuration sidebar"""
        st.sidebar.header("🛠️ Model Configuration")
        provider = st.sidebar.selectbox(
            "AI Provider",
            ["Ollama", "DeepSeek", "Gemini", "ChatGPT", "Anthropic",
             "Claude", "Mistral", "Groq", "HuggingFace"],
            index=0
        )

        saved_settings = self.load_settings(provider)
        settings = {}

        if provider != "Ollama":
            settings["api_key"] = st.sidebar.text_input(
                f"{provider} API Key",
                value=saved_settings.get("api_key", ""),
                type="password"
            )

        if provider == "Ollama":
            settings["host"] = st.sidebar.text_input(
                "API URL",
                value=saved_settings.get("host", "http://127.0.0.1:11434")
            )

        if provider == "HuggingFace":
            settings["endpoint"] = st.sidebar.text_input(
                "Endpoint URL",
                value=saved_settings.get("endpoint", "https://api-inference.huggingface.co/models")
            )

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("💾 Save Settings"):
                self.save_settings(provider, settings)
        with col2:
            if st.button("🔄 Reset"):
                if Path(SETTINGS_FILE).exists():
                    Path(SETTINGS_FILE).unlink()
                    st.success("Settings reset!")

        # Initialize model client
        model_client = self.create_model_client(provider, settings)
        if not model_client:
            return None, None

        # Model selection for providers that support multiple models
        if provider != "HuggingFace":
            try:
                with st.spinner(f"Loading {provider} models..."):
                    models = model_client.list_models()
                    st.session_state.selected_model = st.sidebar.selectbox(
                        "Select Model",
                        models,
                        key=f"{provider}_model_select"
                    )
            except Exception as e:
                logger.error(f"Model load error: {e}")
                st.error(f"Failed to load models: {e}")
                return None, None

        return model_client, st.session_state.selected_model

    def create_model_client(self, provider: str, settings: dict):
        """Instantiate the appropriate model client"""
        try:
            clients = {
                "Ollama": OllamaClient(host=settings.get("host")),
                "DeepSeek": DeepSeekClient(api_key=settings.get("api_key")),
                "Gemini": GeminiClient(api_key=settings.get("api_key")),
                "ChatGPT": ChatGPTClient(api_key=settings.get("api_key")),
                "Anthropic": AnthropicClient(api_key=settings.get("api_key")),
                "Claude": ClaudeClient(api_key=settings.get("api_key")),
                "Mistral": MistralClient(api_key=settings.get("api_key")),
                "Groq": GroqClient(api_key=settings.get("api_key")),
                "HuggingFace": HuggingFaceClient(
                    api_key=settings.get("api_key"),
                    endpoint=settings.get("endpoint")
                )
            }
            return clients[provider]
        except Exception as e:
            logger.error(f"Client init error: {e}")
            st.error(f"{provider} initialization failed")
            return None

    def dataset_structure_interface(self):
        """Render dataset structure configuration UI"""
        st.header("📐 Dataset Structure")

        # Dynamic field management
        cols = st.columns([4, 1])
        with cols[0]:
            st.subheader("Fields & Examples")
        with cols[1]:
            if st.button("➕ Add Field"):
                st.session_state.fields.append(f"field_{len(st.session_state.fields) + 1}")
                st.session_state.examples.append("")

        for i in range(len(st.session_state.fields)):
            cols = st.columns([1, 3, 1])
            with cols[0]:
                st.session_state.fields[i] = st.text_input(
                    f"Field {i + 1}",
                    value=st.session_state.fields[i],
                    key=f"field_{i}"
                )
            with cols[1]:
                st.session_state.examples[i] = st.text_input(
                    f"Example {i + 1}",
                    value=st.session_state.examples[i],
                    key=f"example_{i}"
                )
            with cols[2]:
                if i > 0 and st.button("🗑️", key=f"del_{i}"):
                    del st.session_state.fields[i]
                    del st.session_state.examples[i]
                    st.experimental_rerun()

        # Separator selection
        st.subheader("Formatting")
        cols = st.columns(2)
        with cols[0]:
            st.session_state.data_separator = st.selectbox(
                "Data Wrapper", ['"', "'"], index=0
            )
        with cols[1]:
            st.session_state.section_separator = st.selectbox(
                "Field Separator", ["|", ":", "-", "~"], index=0
            )

    def generation_interface(self, model_client, model_name: str):
        """Handle dataset generation workflow"""
        st.header("⚡ Generation Parameters")

        cols = st.columns([2, 1])
        with cols[0]:
            num_entries = st.slider(
                "Number of Entries", 50, 5000, 500, 50,
                help="Select the desired number of dataset entries"
            )
        with cols[1]:
            st.write("\n")
            if st.button("✨ Generate Dataset", use_container_width=True):
                self.generate_dataset(model_client, model_name, num_entries)

        # Only show generated dataset if it exists
        if st.session_state.is_generated:
            self.post_generation_interface()

    def generate_dataset(self, model_client, model_name: str, num_entries: int):
        """Execute dataset generation process"""
        try:
            heading = st.session_state.section_separator.join(
                f'{st.session_state.data_separator}{f}{st.session_state.data_separator}'
                for f in st.session_state.fields
            )

            davyd = Davyd(
                num_entries=num_entries,
                model_client=model_client,
                model_name=model_name,
                dataset_manager=self.dataset_manager,
                section_separator=st.session_state.section_separator,
                data_separator=st.session_state.data_separator
            )

            with st.spinner("🚀 Generating dataset..."):
                davyd.generate_dataset(heading, st.session_state.examples)
                st.session_state.dataset = davyd.dataset

                # Save to temporary file
                st.session_state.temp_filename = self.dataset_manager.get_temp_filename("dataset")
                self.dataset_manager.save_csv_file(
                    pd.DataFrame(st.session_state.dataset),
                    st.session_state.temp_filename
                )
                st.session_state.is_generated = True  # Mark dataset as generated
                st.success("✅ Generation complete!")

        except Exception as e:
            logger.error(f"Generation error: {e}")
            st.error(f"Generation failed: {str(e)}")
            st.session_state.dataset = []
            st.session_state.is_generated = False

    def post_generation_interface(self):
        """Display generated dataset and management tools"""
        st.header("📊 Generated Dataset")

        # Data Editor
        df = pd.DataFrame(st.session_state.dataset)
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            height=600,
            key="dataset_editor"
        )

        # Action buttons
        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("🔄 Update Dataset"):
                st.session_state.dataset = edited_df.to_dict("records")
        with cols[1]:
            if st.button("📦 Archive Dataset"):
                self.archive_current_dataset()

        # Data Quality Dashboard
        st.header("📈 Data Quality Insights")
        with st.expander("Quality Metrics"):
            if st.session_state.dataset:
                self.visualization.generate_dashboard(pd.DataFrame(st.session_state.dataset))
            else:
                st.warning("No dataset available for analysis")

        # Dataset Management
        st.header("🗂️ Generated Datasets")
        self.generated_datasets_interface()

    def archive_current_dataset(self):
        """Archive the current dataset"""
        if st.session_state.temp_filename:
            try:
                self.dataset_manager.archive_dataset(st.session_state.temp_filename)
                st.success("Dataset archived successfully!")
            except Exception as e:
                st.error(f"Archiving failed: {str(e)}")
        else:
            st.warning("No dataset to archive")

    def generated_datasets_interface(self):
        """Display all generated datasets (active, archived, merged)"""
        try:
            # List active, archived, and merged datasets
            active_datasets = self.dataset_manager.list_datasets()
            archived_datasets = self.dataset_manager.list_archived_datasets()
            merged_datasets = self.dataset_manager.list_merged_datasets()

            if not active_datasets and not archived_datasets and not merged_datasets:
                st.warning("No generated datasets available.")
                return

            # Combine all datasets for selection
            all_datasets = [
                (dataset, "📦 Active") for dataset in active_datasets
            ] + [
                (dataset, "📁 Archive") for dataset in archived_datasets
            ] + [
                (dataset, "🔄 Merge") for dataset in merged_datasets
            ]

            selected = st.selectbox("Select Dataset", all_datasets, format_func=lambda x: f"{x[0]} ({x[1]})", key="generated_datasets_select")

            # Determine if the selected dataset is archived, merged, or active
            dataset_name, dataset_type = selected
            st.session_state.selected_dataset = (dataset_name, dataset_type)

            # Show the selected dataset
            if dataset_name:
                self.show_selected_dataset(dataset_name, dataset_type)

            cols = st.columns(4)
            with cols[0]:  # Delete button
                if st.button("🗑️ Delete Dataset", key="delete_dataset_button"):
                    if dataset_type == "📁 Archive":
                        dataset_path = os.path.join(self.dataset_manager.archive_dir, dataset_name)
                    elif dataset_type == "🔄 Merge":
                        dataset_path = os.path.join(self.dataset_manager.merged_dir, dataset_name)
                    else:
                        dataset_path = os.path.join(self.dataset_manager.temp_dir, dataset_name)

                    if self.dataset_manager.delete_dataset(dataset_path):
                        st.success(f"Dataset '{dataset_name}' deleted successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete dataset '{dataset_name}'")

            with cols[1]:  # Download button
                if st.button("📥 Download", key="download_dataset_button"):
                    self.download_dataset(dataset_name)

            with cols[2]:  # Restore button (only for archived datasets)
                if dataset_type == "📁 Archive":
                    if st.button("📤 Restore", key="restore_dataset_button"):
                        if self.dataset_manager.restore_archived_dataset(dataset_name):
                            st.success(f"Dataset '{dataset_name}' restored successfully!")
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to restore dataset '{dataset_name}'")
                else:
                    st.write("")  # Placeholder for non-archived datasets

            # Dataset merging
            st.subheader("Merge Datasets")
            merge_targets = st.multiselect("Select datasets to merge", [ds[0] for ds in all_datasets], key="merge_datasets_select")
            merged_name = st.text_input("Merged dataset name", "merged_dataset.csv", key="merged_name_input")
            if st.button("🔀 Merge Selected", key="merge_datasets_button"):
                merged_path = os.path.join(self.dataset_manager.merged_dir, merged_name)
                self.dataset_manager.merge_datasets(merge_targets, merged_name)
                st.success("Datasets merged successfully!")
                st.session_state.merged_datasets.append(merged_name)

                # Option to archive the merged dataset
                if st.button("📦 Archive Merged Dataset", key="archive_merged_dataset_button"):
                    self.dataset_manager.archive_dataset(merged_path)
                    st.success("Merged dataset archived successfully!")

        except Exception as e:
            st.error(f"Dataset management error: {str(e)}")

    def show_selected_dataset(self, dataset_name: str, dataset_type: str):
        """Show the selected dataset"""
        try:
            # Load the selected dataset
            if dataset_type == "📁 Archive":
                df = self.dataset_manager.load_archived_dataset(dataset_name)
            elif dataset_type == "🔄 Merge":
                df = self.dataset_manager.load_merged_dataset(dataset_name)
            else:
                df = self.dataset_manager.load_dataset(dataset_name)

            st.header(f"Dataset: {dataset_name} ({dataset_type})")
            st.dataframe(df, use_container_width=True, height=600)

            # Data Quality Dashboard
            st.header("📈 Data Quality Insights")
            with st.expander("Quality Metrics"):
                if not df.empty:
                    self.visualization.generate_dashboard(df)
                else:
                    st.warning("No dataset available for analysis")

        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")

    def download_dataset(self, dataset_name: str):
        """Handle dataset download for active, archived, and merged datasets"""
        try:
            # Determine the type of dataset
            if dataset_name in self.dataset_manager.list_archived_datasets():
                df = self.dataset_manager.load_archived_dataset(dataset_name)
            elif dataset_name in self.dataset_manager.list_merged_datasets():
                df = self.dataset_manager.load_merged_dataset(dataset_name)
            else:
                df = self.dataset_manager.load_dataset(dataset_name)

            format = st.selectbox("Download Format", ["csv", "json", "xlsx"], key="download_format_select")
            self.dataset_manager.download_dataset(df, dataset_name, format)
        except Exception as e:
            st.error(f"Download failed: {str(e)}")

    def restore_dataset(self, dataset_name: str):
        """Restore an archived dataset"""
        try:
            if self.dataset_manager.restore_archived_dataset(dataset_name):
                st.success(f"Dataset '{dataset_name}' restored successfully!")
            else:
                st.error(f"Failed to restore dataset '{dataset_name}'")
        except Exception as e:
            st.error(f"Restore failed: {str(e)}")

    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="DAVYD - Dataset Generator",
            page_icon="🔥",
            layout="wide"
        )
        st.title("🔥 DAVYD - AI-Powered Dataset Generator")

        model_client, model_name = self.model_config_sidebar()
        if not model_client or not model_name:
            return

        self.dataset_structure_interface()
        self.generation_interface(model_client, model_name)

        # Add generated datasets management to the UI
        if self.dataset_manager.list_datasets() or self.dataset_manager.list_archived_datasets() or self.dataset_manager.list_merged_datasets():
            self.generated_datasets_interface()


if __name__ == "__main__":
    ui = DatasetUI()
    ui.run()