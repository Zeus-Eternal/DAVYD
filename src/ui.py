# src/ui.py

import sys
from pathlib import Path
import streamlit as st
from davyd import Davyd  # Ensure this import is correct
import pandas as pd
import logging
from ollama_client import OllamaClient
from autogen.scheduler import Scheduler
from autogen.field_suggester import FieldSuggester
from autogen.template_manager import TemplateManager
from autogen.data_validator import DataValidator
from autogen.intelligent_suggester import IntelligentSuggester
from autogen.visualization import Visualization

# Add 'src' directory to Python path
sys.path.append(str(Path(__file__).parent.resolve()))

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)

def display_data_quality_metrics(dataset: list, visualization: Visualization):
    """
    Display data quality metrics for the dataset.
    """
    if not dataset:
        st.warning("No data available to display metrics.")
        return

    df = pd.DataFrame(dataset)
    
    st.header("📊 Data Quality Insights")
    
    # Generate Dashboard Visualizations
    visualization.generate_dashboard(df)

def main():
    st.set_page_config(page_title="🔥 Real Dataset Generator", layout="wide")
    st.title("🔥 Real Dataset Generator")

    st.sidebar.header("Configuration")

    # Sidebar: Ollama API Configuration
    st.sidebar.subheader("Ollama API Configuration")
    ollama_host = st.sidebar.text_input(
        "Ollama API URL",
        value="http://127.0.0.1:11434",
        help="Enter the Ollama API URL."
    )

    # Initialize OllamaClient
    ollama_client = OllamaClient(host=ollama_host)

    # Initialize Scheduler
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = Scheduler()
    scheduler = st.session_state.scheduler

    # Initialize Autogen Components
    field_suggester = FieldSuggester(ollama_client)
    template_manager = TemplateManager()
    data_validator = DataValidator()
    intelligent_suggester = IntelligentSuggester(ollama_client)
    visualization = Visualization()

    # Fetch available models
    with st.spinner("Fetching Ollama models..."):
        try:
            ollama_models_response = ollama_client.list_models()
            ollama_models = [model['name'] if isinstance(model, dict) else model for model in ollama_models_response]
        except Exception as e:
            logging.error(f"Failed to fetch Ollama models: {e}")
            ollama_models = ["llama3.2:latest"]  # Fallback default model

    if not ollama_models:
        st.sidebar.error("No models found or failed to fetch models from Ollama API.")
        ollama_models = ["llama3.2:latest"]  # Fallback default model

    selected_model = st.sidebar.selectbox("Select Ollama Model", ollama_models)

    # Sidebar: Manage Models
    with st.sidebar.expander("Manage Models"):
        new_model = st.text_input("Add New Model", key="add_model")
        add_model_button = st.button("Add Model")
        if add_model_button:
            if new_model and new_model not in ollama_models:
                try:
                    st.info(f"Pulling model '{new_model}'...")
                    ollama_client.pull(new_model)
                    ollama_models.append(new_model)
                    st.success(f"Model '{new_model}' added.")
                except Exception as e:
                    st.error(f"Failed to add model '{new_model}': {e}")
            elif new_model in ollama_models:
                st.warning("Model already exists.")
            else:
                st.warning("Please enter a valid model name.")

        if len(ollama_models) > 0:
            remove_model = st.selectbox("Remove Model", options=[""] + ollama_models, key="remove_model_select")
            remove_model_button = st.button("Remove Selected Model")
            if remove_model_button:
                if remove_model and remove_model in ollama_models:
                    try:
                        ollama_client.delete(remove_model)
                        ollama_models.remove(remove_model)
                        st.success(f"Model '{remove_model}' removed.")
                        if selected_model == remove_model:
                            selected_model = ollama_models[0] if ollama_models else None
                    except Exception as e:
                        st.error(f"Failed to remove model '{remove_model}': {e}")
                elif remove_model == "":
                    st.warning("No model selected to remove.")
                else:
                    st.warning("Model not found.")

    # Step 1: Define Dataset Structure
    st.header("1. Define Dataset Structure")
    
    # Initialize session state for dynamic fields
    if 'fields' not in st.session_state:
        st.session_state.fields = ["text", "intent", "sentiment", "sentiment_polarity", "tone", "category", "keywords"]
    if 'examples' not in st.session_state:
        st.session_state.examples = [
            '"Hi there!"',
            '"greeting"',
            '"positive"',
            '0.9',
            '"friendly"',
            '"interaction"',
            '"hi" "hello" "welcome"'
        ]

    # Template Selection Section
    st.subheader("Template-Based Setup")
    templates = template_manager.load_templates()
    selected_template = st.selectbox("Select a Template", ["None"] + list(templates.keys()))
    if selected_template != "None":
        template_structure = templates[selected_template]
        st.session_state.fields = [field['name'] for field in template_structure['fields']]
        st.session_state.examples = [""] * len(template_structure['fields'])  # Optionally, load examples if available
        st.success(f"Template '{selected_template}' loaded successfully.")

    # Display existing fields and examples with input boxes
    for i in range(len(st.session_state.fields)):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.fields[i] = st.text_input(
                f"Field {i + 1} Name", 
                value=st.session_state.fields[i], 
                key=f"field_name_{i}"
            )
        with col2:
            st.session_state.examples[i] = st.text_input(
                f"Example for '{st.session_state.fields[i]}'", 
                value=st.session_state.examples[i], 
                key=f"example_text_{i}"
            )

    # Button to add new field-example pair
    if st.button("➕ Add New Field"):
        st.session_state.fields.append(f"field_{len(st.session_state.fields) + 1}")
        st.session_state.examples.append("")

    # Button to remove a field
    if len(st.session_state.fields) > 1:
        remove_field = st.selectbox("Select Field to Remove", options=st.session_state.fields, key="remove_field_select")
        if st.button("➖ Remove Selected Field"):
            if remove_field in st.session_state.fields:
                index = st.session_state.fields.index(remove_field)
                st.session_state.fields.pop(index)
                st.session_state.examples.pop(index)
                st.success(f"Field '{remove_field}' removed successfully.")
            else:
                st.warning("Selected field not found.")
    else:
        st.warning("At least one field is required.")

    # Auto-Suggest Fields Section
    st.subheader("Auto-Suggest Fields")
    user_description = st.text_area("Describe your dataset structure:", height=100, help="Provide a brief description to auto-suggest fields and types.")
    if st.button("🔍 Suggest Fields"):
        if user_description.strip():
            suggestions = field_suggester.suggest_fields(user_description)
            if suggestions:
                for suggestion in suggestions:
                    st.session_state.fields.append(suggestion['name'])
                    st.session_state.examples.append("")  # Optionally, prompt for example
                st.success("Fields suggested and added successfully.")
            else:
                st.warning("No suggestions available. Please refine your description.")
        else:
            st.warning("Please enter a description to get field suggestions.")

    # Intelligent Data Suggestions Section
    st.subheader("Intelligent Data Suggestions")

    # Dropdown to select the field for suggestions
    field_to_suggest = st.selectbox("Select Field to Suggest Values For", st.session_state.fields)

    # Button to trigger value suggestions
    if field_to_suggest:
        if st.button("💡 Suggest Values"):
            try:
                # Ensure valid examples are provided for context
                if not st.session_state.examples:
                    st.warning("No examples available to generate suggestions.")
                else:
                    # Generate suggestions using IntelligentSuggester
                    suggestions = intelligent_suggester.suggest_field_values(
                        st.session_state.examples, field_to_suggest
                    )
                    if suggestions:
                        # Handle updates based on field type
                        field_index = st.session_state.fields.index(field_to_suggest)
                        if field_to_suggest.lower() == "keywords":
                            # Format keywords as space-separated strings
                            st.session_state.examples[field_index] = '"' + '" "'.join(suggestions) + '"'
                        else:
                            # General case: comma-separated values
                            st.session_state.examples[field_index] = ', '.join(suggestions)
                        st.success(f"Suggested values for '{field_to_suggest}' have been updated.")
                    else:
                        st.warning("No suggestions available for the selected field.")
            except AttributeError as e:
                st.error(f"AttributeError: {e}")
                logging.error(f"Error in IntelligentSuggester.suggest_field_values: {e}")
            except Exception as ex:
                st.error(f"An unexpected error occurred: {ex}")
                logging.error(f"Unexpected error in suggesting values: {ex}")

    # Generate heading and example rows
    heading = "|".join([f'"{field}"' for field in st.session_state.fields])
    example_rows = [heading] + ["|".join(st.session_state.examples)]

    # Live Preview Area
    st.subheader("Live Preview of Example Rows")
    preview_text = "\n".join(example_rows)
    st.text_area("Example Rows Preview", value=preview_text, height=200, disabled=True)

    # Step 2: Generation Parameters
    st.header("2. Generation Parameters")
    num_entries = st.slider("Number of Entries", min_value=50, max_value=1000, value=150, step=50)

    # Step 3: Advanced Settings (Optional)
    with st.expander("Advanced Settings 🔍"):
        st.write("Automated settings are managed through Autogen features.")

    # Initialize Davyd
    generator = Davyd(
        num_entries=num_entries, 
        ollama_host=ollama_host, 
        ollama_model=selected_model
    )

    # Define the scheduled task
    def scheduled_dataset_generation():
        """
        Function to generate the dataset automatically.
        """
        try:
            generator.generate_dataset(heading, st.session_state.examples)
            st.session_state["dataset"] = generator.dataset
            logging.info("Scheduled dataset generation completed successfully.")
        except Exception as e:
            logging.error(f"Scheduled dataset generation failed: {e}")

    # Generate Dataset Button
    if st.button("✨ Generate Dataset"):
        with st.spinner("Generating dataset..."):
            try:
                generator.generate_dataset(heading, st.session_state.examples)
                st.success("🔥 Dataset generation complete!")
                st.session_state["dataset"] = generator.dataset
            except ValueError as ve:
                st.error(f"❌ Value Error: {ve}")
                logging.error(f"Value Error during generation: {ve}")
            except Exception as e:
                st.error(f"❌ Error during generation: {e}")
                logging.error(f"Error during generation: {e}")

    # Display and Modify Dataset
    if "dataset" in st.session_state:
        dataset = st.session_state["dataset"]
        df = pd.DataFrame(dataset)
        st.header("3. Preview & Modify Dataset")
        st.write("**Fields:**")
        st.write(", ".join(st.session_state.fields))  # Show title fields at the top
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        )

        if st.button("💾 Save & Export"):
            output_format = st.selectbox("Output Format", ["csv", "json"], index=0)
            filename = f"generated_dataset.{output_format}"
            try:
                # Define fields_info based on current fields and types
                fields_info = [{'name': field, 'type': 'string'} for field in st.session_state.fields]  # Modify types as needed
                errors = data_validator.validate_dataset(edited_df.to_dict(orient='records'), fields_info)
                if errors:
                    st.error("Dataset validation failed:")
                    for error in errors:
                        st.error(error)
                else:
                    generator.dataset = edited_df.to_dict(orient='records')
                    generator.save_dataset(filename, output_format=output_format)
                    st.success(f"✅ Dataset saved as {filename}!")
                    with open(filename, 'rb') as file:
                        st.download_button(
                            label="📥 Download Dataset",
                            data=file,
                            file_name=filename,
                            mime='application/octet-stream'
                        )
            except Exception as e:
                st.error(f"❌ Error saving dataset: {e}")

        # Display Data Quality Metrics
        display_data_quality_metrics(generator.dataset, visualization)

        # Export Template Section
        st.subheader("Export Template")
        export_template_name = st.selectbox("Select Template to Export", ["None"] + list(template_manager.load_templates().keys()))
        export_path = st.text_input("Export Path (e.g., /path/to/export/template.json)")
        if st.button("📤 Export Template"):
            if export_template_name != "None" and export_path:
                template_manager.export_template(export_template_name, export_path)
                st.success(f"Template '{export_template_name}' exported successfully to {export_path}.")
            else:
                st.warning("Please select a template and provide a valid export path.")

        # Schedule Dataset Generation
        st.header("4. Schedule Dataset Generation")
        with st.form("scheduler_form"):
            schedule_interval = st.selectbox("Select Schedule Interval", ["None", "Hourly", "Daily"])
            time_of_day = st.text_input(
                "Select Time of Day for Daily Schedule (HH:MM)", 
                value="10:00", 
                help="Format: HH:MM (24-hour)"
            )
            submit_scheduler = st.form_submit_button("🕒 Schedule Generation")
            if submit_scheduler:
                if schedule_interval != "None":
                    try:
                        # Schedule the task
                        scheduler.schedule_task(
                            scheduled_dataset_generation, 
                            interval=schedule_interval.lower(), 
                            time_of_day=time_of_day
                        )
                        scheduler.start()
                        st.success(f"✅ Dataset generation scheduled {schedule_interval.lower()}ly at {time_of_day}.")
                    except ValueError as ve:
                        st.error(f"❌ Scheduling Error: {ve}")
                else:
                    st.info("No scheduling selected.")

if __name__ == "__main__":
    main()
