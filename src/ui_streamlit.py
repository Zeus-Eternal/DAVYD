import os
import json
import logging
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
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
MAX_DATASET_SIZE = 10_000  # Maximum allowed dataset entries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetUI:
    """Enhanced dataset generator UI with improved functionality"""

    def __init__(self):
        self.dataset_manager = DatasetManager(TEMP_DIR, ARCHIVE_DIR, MERGED_DIR)
        self.visualization = Visualization()
        self.initialize_session_state()
        self._setup_page_config()

    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="DAVYD - Dataset Generator",
            page_icon="🔥",
            layout="wide",
            menu_items={
                'Get Help': 'https://github.com/yourrepo/davyd',
                'Report a bug': "https://github.com/yourrepo/davyd/issues",
                'About': "# DAVYD - AI-Powered Dataset Generator"
            }
        )

    def initialize_session_state(self):
        """Initialize and validate session state variables"""
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
            'is_generated': False,
            'merged_datasets': [],
            'selected_dataset': None,
            'last_generated': None
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Validate existing dataset structure
        if st.session_state.dataset and not self._validate_dataset_structure():
            st.session_state.dataset = []
            st.session_state.is_generated = False

    def _validate_dataset_structure(self) -> bool:
        """Validate the structure of existing dataset in session state"""
        try:
            if not st.session_state.dataset:
                return False
            
            # Check if all entries have the same keys
            first_keys = set(st.session_state.dataset[0].keys())
            for entry in st.session_state.dataset[1:]:
                if set(entry.keys()) != first_keys:
                    return False
            return True
        except Exception as e:
            logger.error(f"Dataset validation error: {e}")
            return False

    def load_settings(self, provider: str) -> dict:
        """Load settings with improved error handling"""
        try:
            if not Path(SETTINGS_FILE).exists():
                return {}

            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                return settings.get(provider, {})
                
        except json.JSONDecodeError as je:
            logger.error(f"Corrupted settings file: {je}")
            st.warning("Settings file is corrupted. Resetting to defaults.")
            Path(SETTINGS_FILE).unlink(missing_ok=True)
            return {}
        except Exception as e:
            logger.error(f"Settings load error: {e}")
            return {}

    def save_settings(self, provider: str, settings: dict):
        """Save settings with atomic write operation"""
        try:
            settings_dir = Path(SETTINGS_FILE).parent
            settings_dir.mkdir(exist_ok=True, parents=True)
            
            # Create temp file for atomic write
            temp_file = f"{SETTINGS_FILE}.tmp"
            
            all_settings = self.load_settings("__all__") or {}
            all_settings[provider] = settings
            
            with open(temp_file, "w") as f:
                json.dump(all_settings, f, indent=4)
            
            # Atomic rename
            Path(temp_file).replace(SETTINGS_FILE)
            st.success(f"{provider} settings saved!")
            
        except Exception as e:
            logger.error(f"Settings save error: {e}")
            st.error("Failed to save settings. Check logs for details.")

    def model_config_sidebar(self) -> Tuple[Optional[BaseModelClient], Optional[str]]:
        """Enhanced model configuration with connection testing"""
        st.sidebar.header("🛠️ Model Configuration")
        
        provider = st.sidebar.selectbox(
            "AI Provider",
            ["Ollama", "DeepSeek", "Gemini", "ChatGPT", "Anthropic",
             "Claude", "Mistral", "Groq", "HuggingFace"],
            index=0
        )

        saved_settings = self.load_settings(provider)
        settings = {}

        # Provider-specific settings
        if provider != "Ollama":
            settings["api_key"] = st.sidebar.text_input(
                f"{provider} API Key",
                value=saved_settings.get("api_key", ""),
                type="password",
                help="Required for API access"
            )
        else:
            settings["host"] = st.sidebar.text_input(
                "API URL",
                value=saved_settings.get("host", "http://127.0.0.1:11434"),
                help="Ollama server endpoint"
            )

        if provider == "HuggingFace":
            settings["endpoint"] = st.sidebar.text_input(
                "Endpoint URL",
                value=saved_settings.get("endpoint", "https://api-inference.huggingface.co/models"),
                help="HuggingFace inference endpoint"
            )

        # Settings management
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("💾 Save Settings", help="Save current configuration"):
                self.save_settings(provider, settings)
        with col2:
            if st.button("🔄 Reset", help="Reset to default settings"):
                Path(SETTINGS_FILE).unlink(missing_ok=True)
                st.success("Settings reset to defaults!")
                st.experimental_rerun()

        # Initialize client with connection test
        try:
            model_client = self.create_model_client(provider, settings)
            if not model_client:
                return None, None

            # Model selection with enhanced handling
            if provider != "HuggingFace":
                with st.spinner(f"Loading {provider} models..."):
                    try:
                        models = model_client.list_models()
                        if not models:
                            st.error(f"No models available for {provider}")
                            return None, None
                            
                        st.session_state.selected_model = st.sidebar.selectbox(
                            "Select Model",
                            models,
                            key=f"{provider}_model_select",
                            help="Select model to use for generation"
                        )
                        
                        # Test model connection
                        if st.sidebar.button("🧪 Test Connection", help="Verify model accessibility"):
                            with st.spinner("Testing connection..."):
                                if model_client.test_connection():
                                    st.success("Connection successful!")
                                else:
                                    st.error("Connection failed")
                    except Exception as e:
                        st.error(f"Model loading failed: {str(e)}")
                        logger.exception("Model loading error")
                        return None, None

            return model_client, st.session_state.selected_model
            
        except Exception as e:
            st.error(f"Client initialization failed: {str(e)}")
            logger.exception("Client initialization error")
            return None, None

    def create_model_client(self, provider: str, settings: dict) -> Optional[BaseModelClient]:
        """Create model client with enhanced validation"""
        try:
            # Validate required settings
            if provider != "Ollama" and not settings.get("api_key"):
                st.error(f"{provider} requires an API key")
                return None

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
            
            client = clients[provider]
            
            # Verify basic connectivity
            if not client.test_connection():
                st.error(f"Could not connect to {provider} service")
                return None
                
            return client
            
        except Exception as e:
            logger.error(f"Client init error for {provider}: {e}")
            st.error(f"{provider} initialization failed")
            return None

    def dataset_structure_interface(self):
        """Enhanced dataset structure configuration"""
        st.header("📐 Dataset Structure")
        
        # Field management with improved UX
        cols = st.columns([4, 1])
        with cols[0]:
            st.subheader("Fields & Examples")
            st.caption("Define the structure of your dataset")
        with cols[1]:
            if st.button("➕ Add Field", help="Add new field to dataset structure"):
                st.session_state.fields.append(f"field_{len(st.session_state.fields) + 1}")
                st.session_state.examples.append("")
                st.experimental_rerun()

        # Dynamic field editing
        for i in range(len(st.session_state.fields)):
            cols = st.columns([1, 3, 1])
            with cols[0]:
                st.session_state.fields[i] = st.text_input(
                    f"Field {i + 1}",
                    value=st.session_state.fields[i],
                    key=f"field_{i}",
                    help="Field name (e.g., 'text', 'sentiment')"
                )
            with cols[1]:
                st.session_state.examples[i] = st.text_input(
                    f"Example {i + 1}",
                    value=st.session_state.examples[i],
                    key=f"example_{i}",
                    help="Example value for this field"
                )
            with cols[2]:
                if i > 0 and st.button("🗑️", key=f"del_{i}", help="Remove this field"):
                    del st.session_state.fields[i]
                    del st.session_state.examples[i]
                    st.experimental_rerun()

        # Formatting options
        st.subheader("Formatting Options")
        cols = st.columns(2)
        with cols[0]:
            st.session_state.data_separator = st.selectbox(
                "Data Wrapper", ['"', "'"], index=0,
                help="Character to wrap field values"
            )
        with cols[1]:
            st.session_state.section_separator = st.selectbox(
                "Field Separator", ["|", ":", "-", "~"], index=0,
                help="Character to separate fields"
            )

    def generation_interface(self, model_client, model_name: str):
        """Enhanced generation interface with progress tracking"""
        st.header("⚡ Generation Parameters")
        
        # Generation controls
        cols = st.columns([2, 1])
        with cols[0]:
            num_entries = st.slider(
                "Number of Entries", 
                50, MAX_DATASET_SIZE, 500, 50,
                help=f"Select number of entries to generate (max {MAX_DATASET_SIZE})"
            )
        with cols[1]:
            if st.button(
                "✨ Generate Dataset", 
                use_container_width=True,
                disabled=not (model_client and model_name),
                help="Generate dataset with current parameters"
            ):
                with st.spinner("Preparing generation..."):
                    self.generate_dataset(model_client, model_name, num_entries)

        # Display generated dataset if available
        if st.session_state.is_generated:
            self.post_generation_interface()

    def generate_dataset(self, model_client, model_name: str, num_entries: int):
        """Enhanced dataset generation with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Validate inputs
            if not model_client or not model_name:
                raise ValueError("Model client and name must be specified")
                
            if num_entries <= 0 or num_entries > MAX_DATASET_SIZE:
                raise ValueError(f"Number of entries must be between 1 and {MAX_DATASET_SIZE}")

            status_text.text("Initializing dataset generation...")
            progress_bar.progress(5)

            # Create heading from fields
            heading = st.session_state.section_separator.join(
                f'{st.session_state.data_separator}{f}{st.session_state.data_separator}'
                for f in st.session_state.fields
            )

            # Initialize Davyd generator
            davyd = Davyd(
                num_entries=num_entries,
                model_client=model_client,
                model_name=model_name,
                dataset_manager=self.dataset_manager,
                section_separator=st.session_state.section_separator,
                data_separator=st.session_state.data_separator
            )

            status_text.text("Generating dataset entries...")
            progress_bar.progress(30)

            # Generate dataset
            davyd.generate_dataset(heading, st.session_state.examples)
            
            status_text.text("Processing generated data...")
            progress_bar.progress(70)

            # Store results
            st.session_state.dataset = davyd.dataset
            st.session_state.temp_filename = self.dataset_manager.get_temp_filename("dataset")
            self.dataset_manager.save_csv_file(
                pd.DataFrame(st.session_state.dataset),
                st.session_state.temp_filename
            )
            
            st.session_state.is_generated = True
            st.session_state.last_generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            status_text.text("Finalizing...")
            progress_bar.progress(100)
            st.success("✅ Generation complete!")
            
        except HTTPError as he:
            if he.response.status_code == 401:
                st.error("Authentication failed - please check your API key")
            else:
                st.error(f"API error: {he.response.text}")
            logger.error(f"HTTP error during generation: {he}")
            st.session_state.dataset = []
            st.session_state.is_generated = False
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            logger.error(f"Generation error: {e}", exc_info=True)
            st.session_state.dataset = []
            st.session_state.is_generated = False
        finally:
            progress_bar.empty()
            status_text.empty()

    def post_generation_interface(self):
        """Enhanced post-generation interface with more features"""
        st.header("📊 Generated Dataset")
        
        # Dataset editing and management
        df = pd.DataFrame(st.session_state.dataset)
        
        # Show dataset info
        cols = st.columns([2, 1])
        with cols[0]:
            st.metric("Total Entries", len(df))
        with cols[1]:
            if st.session_state.last_generated:
                st.caption(f"Last generated: {st.session_state.last_generated}")

        # Data editor with improved display
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            height=600,
            key="dataset_editor",
            column_config={
                "text": st.column_config.TextColumn("Text", width="large"),
                "sentiment_polarity": st.column_config.NumberColumn(
                    "Sentiment", 
                    min_value=-1.0, 
                    max_value=1.0,
                    format="%.2f"
                )
            }
        )

        # Action buttons
        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("🔄 Update Dataset", help="Save changes to dataset"):
                st.session_state.dataset = edited_df.to_dict("records")
                st.success("Dataset updated!")
        with cols[1]:
            if st.button("📦 Archive Dataset", help="Move to archived datasets"):
                self.archive_current_dataset()

        # Enhanced data quality dashboard
        st.header("📈 Data Quality Insights")
        with st.expander("🔍 Detailed Analysis", expanded=True):
            if st.session_state.dataset:
                self.visualization.generate_dashboard(pd.DataFrame(st.session_state.dataset))
                self.show_dataset_stats(pd.DataFrame(st.session_state.dataset))
            else:
                st.warning("No dataset available for analysis")

        # Dataset management section
        st.header("🗂️ Dataset Management")
        self.generated_datasets_interface()

    def show_dataset_stats(self, df: pd.DataFrame):
        """Display comprehensive dataset statistics"""
        with st.expander("📊 Dataset Statistics", expanded=False):
            tabs = st.tabs(["Overview", "Field Stats", "Quality Checks"])
            
            with tabs[0]:
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Total Entries", len(df))
                with cols[1]:
                    st.metric("Columns", len(df.columns))
                with cols[2]:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                st.write("**Column Types:**")
                st.write(df.dtypes.astype(str))
            
            with tabs[1]:
                for col in df.columns:
                    with st.expander(f"Field: {col}"):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            st.write(f"**Statistics for {col}:**")
                            st.write(df[col].describe())
                            st.bar_chart(df[col].value_counts())
                        else:
                            st.write(f"**Top values for {col}:**")
                            st.write(df[col].value_counts().head(10))
            
            with tabs[2]:
                if "sentiment" in df.columns:
                    st.write("**Sentiment Distribution:**")
                    st.bar_chart(df["sentiment"].value_counts())
                
                if "intent" in df.columns:
                    st.write("**Intent Coverage:**")
                    st.write(f"{len(df['intent'].unique())} unique intents")
                
                # Add more quality checks as needed

    def archive_current_dataset(self):
        """Enhanced archiving with confirmation"""
        if not st.session_state.temp_filename:
            st.warning("No dataset to archive")
            return
            
        if st.checkbox("Confirm archiving", key="confirm_archive"):
            try:
                self.dataset_manager.archive_dataset(st.session_state.temp_filename)
                st.session_state.temp_filename = None
                st.session_state.is_generated = False
                st.success("Dataset archived successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Archiving failed: {str(e)}")
                logger.error(f"Archive error: {e}")

    def generated_datasets_interface(self):
        """Enhanced dataset management interface"""
        try:
            # Get all available datasets
            active_datasets = self.dataset_manager.list_datasets()
            archived_datasets = self.dataset_manager.list_archived_datasets()
            merged_datasets = self.dataset_manager.list_merged_datasets()

            if not active_datasets and not archived_datasets and not merged_datasets:
                st.warning("No generated datasets available.")
                return

            # Create combined list with types and unique IDs
            all_datasets = [
                (name, "📦 Active", f"active_{name}") for name in active_datasets
            ] + [
                (name, "📁 Archive", f"archive_{name}") for name in archived_datasets
            ] + [
                (name, "🔄 Merge", f"merge_{name}") for name in merged_datasets
            ]

            # Create unique key based on dataset list
            select_key = f"dataset_select_{hash(tuple(d[2] for d in all_datasets))}"

            # Dataset selection
            selected = st.selectbox(
                "Select Dataset",
                all_datasets,
                format_func=lambda x: f"{x[0]} ({x[1]})",
                key=select_key
            )

            if selected:
                name, dtype, uid = selected
                st.session_state.selected_dataset = (name, dtype)

                # Show dataset preview
                with st.expander("🔍 Dataset Preview", expanded=True):
                    self.show_selected_dataset(name, dtype)

                # Dataset actions
                cols = st.columns(4)
                with cols[0]:
                    if st.button("🗑️ Delete", key=f"delete_{uid}", help="Permanently delete this dataset"):
                        self._delete_dataset(name, dtype)
                with cols[1]:
                    if st.button("📥 Download", key=f"download_{uid}", help="Download this dataset"):
                        self.download_dataset(name)
                with cols[2]:
                    if dtype == "📁 Archive" and st.button("📤 Restore", key=f"restore_{uid}", 
                                                         help="Move to active datasets"):
                        self._restore_dataset(name)

                # Dataset operations
                st.subheader("🔀 Dataset Operations")
                tab1, tab2 = st.tabs(["Merge Datasets", "Bulk Actions"])

                with tab1:
                    self._show_merge_interface(all_datasets)

                with tab2:
                    self._show_bulk_actions_interface()

        except Exception as e:
            st.error(f"Dataset management error: {str(e)}")
            logger.error(f"Dataset interface error: {e}", exc_info=True)

    def _delete_dataset(self, name: str, dtype: str):
        """Handle dataset deletion with confirmation"""
        if not st.checkbox(f"Confirm deletion of {name}", key=f"confirm_delete_{name}"):
            return
            
        try:
            if dtype == "📁 Archive":
                path = os.path.join(self.dataset_manager.archive_dir, name)
            elif dtype == "🔄 Merge":
                path = os.path.join(self.dataset_manager.merged_dir, name)
            else:
                path = os.path.join(self.dataset_manager.temp_dir, name)

            if self.dataset_manager.delete_dataset(path):
                st.success(f"Dataset '{name}' deleted successfully!")
                st.experimental_rerun()
            else:
                st.error(f"Failed to delete dataset '{name}'")
        except Exception as e:
            st.error(f"Delete operation failed: {str(e)}")
            logger.error(f"Delete error for {name}: {e}")

    def _restore_dataset(self, name: str):
        """Handle dataset restoration"""
        try:
            if self.dataset_manager.restore_archived_dataset(name):
                st.success(f"Dataset '{name}' restored successfully!")
                st.experimental_rerun()
            else:
                st.error(f"Failed to restore dataset '{name}'")
        except Exception as e:
            st.error(f"Restore operation failed: {str(e)}")
            logger.error(f"Restore error for {name}: {e}")

    def _show_merge_interface(self, all_datasets: list):
        """Show dataset merging interface"""
        merge_key = f"merge_select_{hash(tuple(d[2] for d in all_datasets))}"
        merge_targets = st.multiselect(
            "Select datasets to merge",
            [ds[0] for ds in all_datasets],
            key=merge_key,
            help="Select multiple datasets to combine"
        )

        name_key = f"merge_name_{hash(tuple(merge_targets))}"
        merged_name = st.text_input(
            "Merged dataset name",
            "merged_dataset.csv",
            key=name_key,
            help="Name for the new merged dataset"
        )

        if st.button("🔀 Merge Selected", key=f"merge_button_{hash(tuple(merge_targets))}"):
            with st.spinner("Merging datasets..."):
                try:
                    self.dataset_manager.merge_datasets(merge_targets, merged_name)
                    st.session_state.merged_datasets.append(merged_name)
                    st.success("Datasets merged successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Merge failed: {str(e)}")
                    logger.error(f"Merge error: {e}")

    def _show_bulk_actions_interface(self):
        """Show bulk dataset operations"""
        st.warning("Bulk actions affect all datasets of the selected type")
        
        if st.button("🗑️ Delete All Temporary Datasets", 
                    help="Permanently delete all active datasets"):
            if st.checkbox("Confirm bulk deletion", key="confirm_bulk_delete"):
                self._bulk_delete_datasets("temp")

        if st.button("🗃️ Archive All Temporary Datasets",
                    help="Move all active datasets to archive"):
            if st.checkbox("Confirm bulk archive", key="confirm_bulk_archive"):
                self._bulk_archive_datasets()

    def _bulk_delete_datasets(self, dataset_type: str):
        """Handle bulk dataset deletion"""
        try:
            if dataset_type == "temp":
                count = self.dataset_manager.delete_all_temp_datasets()
                st.success(f"Deleted {count} temporary datasets!")
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Bulk delete failed: {str(e)}")
            logger.error(f"Bulk delete error: {e}")

    def _bulk_archive_datasets(self):
        """Handle bulk dataset archiving"""
        try:
            count = self.dataset_manager.archive_all_temp_datasets()
            st.success(f"Archived {count} datasets!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Bulk archive failed: {str(e)}")
            logger.error(f"Bulk archive error: {e}")

    def show_selected_dataset(self, dataset_name: str, dataset_type: str):
        """Display selected dataset with enhanced visualization"""
        try:
            # Load dataset based on type
            if dataset_type == "📁 Archive":
                df = self.dataset_manager.load_archived_dataset(dataset_name)
            elif dataset_type == "🔄 Merge":
                df = self.dataset_manager.load_merged_dataset(dataset_name)
            else:
                df = self.dataset_manager.load_dataset(dataset_name)

            # Show basic info
            cols = st.columns(3)
            with cols[0]:
                st.metric("Entries", len(df))
            with cols[1]:
                st.metric("Columns", len(df.columns))
            with cols[2]:
                st.metric("Size", f"{df.memory_usage().sum() / 1024:.2f} KB")

            # Display dataset
            st.dataframe(df, use_container_width=True, height=400)

            # Quick visualization
            if len(df) > 0:
                self.visualization.generate_dashboard(df)

        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")
            logger.error(f"Dataset load error: {e}")

    def download_dataset(self, dataset_name: str):
        """Enhanced dataset download with format options"""
        try:
            # Determine dataset type and load
            if dataset_name in self.dataset_manager.list_archived_datasets():
                df = self.dataset_manager.load_archived_dataset(dataset_name)
            elif dataset_name in self.dataset_manager.list_merged_datasets():
                df = self.dataset_manager.load_merged_dataset(dataset_name)
            else:
                df = self.dataset_manager.load_dataset(dataset_name)

            # Download options
            format = st.selectbox(
                "Download Format",
                ["csv", "json", "xlsx", "parquet"],
                key="download_format_select",
                help="Select output format"
            )

            # Generate download
            if st.button("⬇️ Prepare Download", key=f"dl_{dataset_name}"):
                with st.spinner("Preparing download..."):
                    self.dataset_manager.download_dataset(df, dataset_name, format)
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            logger.error(f"Download error for {dataset_name}: {e}")

    def run(self):
        """Main application runner"""
        st.title("🔥 DAVYD - AI-Powered Dataset Generator")
        
        # Model configuration
        model_client, model_name = self.model_config_sidebar()
        if not model_client or not model_name:
            st.warning("Please configure a valid model first")
            return

        # Main interface sections
        self.dataset_structure_interface()
        self.generation_interface(model_client, model_name)

        # Show dataset management if datasets exist
        if (self.dataset_manager.list_datasets() or 
            self.dataset_manager.list_archived_datasets() or 
            self.dataset_manager.list_merged_datasets()):
            self.generated_datasets_interface()

if __name__ == "__main__":
    DatasetUI().run()
    