# src/ui.py

import streamlit as st
from davyd import Davyd
import pandas as pd
import logging
from ollama_client import OllamaClient  # Import the updated OllamaClient

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)

def display_data_quality_metrics(dataset: list):
    """
    Display data quality metrics for the dataset.
    """
    if not dataset:
        st.warning("No data available to display metrics.")
        return

    df = pd.DataFrame(dataset)
    
    st.header("📊 Data Quality Insights")
    
    # Sentiment Distribution
    if 'sentiment' in df.columns:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    # Intent Distribution
    if 'intent' in df.columns:
        st.subheader("Intent Distribution")
        intent_counts = df['intent'].value_counts()
        st.bar_chart(intent_counts)
    
    # Sentiment Polarity Statistics
    if 'sentiment_polarity' in df.columns:
        st.subheader("Sentiment Polarity Statistics")
        polarity_mean = pd.to_numeric(df['sentiment_polarity'], errors='coerce').mean()
        polarity_median = pd.to_numeric(df['sentiment_polarity'], errors='coerce').median()
        st.write(f"**Mean Sentiment Polarity:** {polarity_mean:.2f}")
        st.write(f"**Median Sentiment Polarity:** {polarity_median:.2f}")
    
    # Keyword Frequency
    if 'keywords' in df.columns:
        st.subheader("Keyword Frequency")
        # Split keywords by quotes and spaces
        keywords_series = df['keywords'].str.split('" "').explode().str.strip('"')
        keyword_counts = keywords_series.value_counts().head(20)
        st.bar_chart(keyword_counts)

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

    # Fetch available models
    with st.spinner("Fetching Ollama models..."):
        try:
            ollama_models_response = ollama_client.list_models()
            if isinstance(ollama_models_response, list):
                if all(isinstance(model, dict) and 'name' in model for model in ollama_models_response):
                    ollama_models = [model['name'] for model in ollama_models_response]
                elif all(isinstance(model, str) for model in ollama_models_response):
                    ollama_models = ollama_models_response
                else:
                    ollama_models = [str(model) for model in ollama_models_response]
            else:
                ollama_models = []
        except Exception as e:
            logging.error(f"Failed to fetch Ollama models: {e}")
            ollama_models = []

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

        remove_model = st.selectbox("Remove Model", options=[""] + ollama_models, key="remove_model_select")
        if st.button("Remove Selected Model"):
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
    # Set number of fields to 7 to align with example data
    num_fields = st.number_input("Number of Fields", min_value=1, max_value=20, value=7, step=1)
    fields = []
    examples = []

    default_field_names = ["text", "intent", "sentiment", "sentiment_polarity", "tone", "category", "keywords"]
    for i in range(num_fields):
        col1, col2 = st.columns(2)
        with col1:
            default_value = default_field_names[i] if i < len(default_field_names) else f"field_{i+1}"
            field_name = st.text_input(f"Field {i + 1} Name", value=default_value, key=f"field_name_{i + 1}")
            fields.append(field_name.strip())
        with col2:
            example_text = st.text_input(f"Example for {field_name}", value=f"Example {i + 1}", key=f"example_text_{i + 1}")
            examples.append(example_text.strip())

    heading = "|".join([f'"{field}"' for field in fields])

    # Step 2: Provide Example Rows
    st.header("2. Provide Example Rows (Optional)")
    st.write("Enter example rows in the following format:")
    st.code(heading, language='plaintext')
    body_examples_input = st.text_area(
        "Example Rows",
        value="""\
"Hi there!"|greeting|positive|0.9|friendly|interaction|"hi" "hello" "welcome"
"Please draw a mountain landscape."|draw_request|neutral|0.5|instructive|art|"draw" "landscape" "mountain"
"I am feeling sad today."|emotion|negative|0.4|somber|personal|"sad" "unhappy" "down"
"Congratulations on your promotion!"|celebration|positive|0.95|joyful|work|"congratulations" "promotion" "achievement"
""",
        help="Provide example rows for context only. Each row should match the number of fields and be pipe-separated."
    ).strip().split("\n")
    body_examples = [example for example in body_examples_input if example]

    # Step 3: Generation Parameters
    st.header("3. Generation Parameters")
    num_entries = st.slider("Number of Entries", min_value=50, max_value=1000, value=150, step=50)

    # Advanced Settings
    with st.expander("Advanced Settings 🔍"):
        # Removed temperature and max_tokens inputs as they are handled via prompt
        st.write("Temperature and Max Tokens are embedded within the generation prompt.")

    # Initialize Davyd
    generator = Davyd(num_entries=num_entries, ollama_host=ollama_host, ollama_model=selected_model)

    # Generate Dataset Button
    if st.button("✨ Generate Dataset"):
        with st.spinner("Generating dataset..."):
            try:
                generator.generate_dataset(heading, body_examples)
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
        st.header("4. Preview & Modify Dataset")
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
        display_data_quality_metrics(generator.dataset)

if __name__ == "__main__":
    main()
