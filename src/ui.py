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
            ollama_models = [model['name'] if isinstance(model, dict) else model for model in ollama_models_response]
        except Exception as e:
            logging.error(f"Failed to fetch Ollama models: {e}")
            ollama_models = ["llama3.2:latest"]  # Fallback default model

    selected_model = st.sidebar.selectbox("Select Ollama Model", ollama_models)

    # Step 1: Define Dataset Structure
    st.header("1. Define Dataset Structure")

    # Manage dynamic fields
    if "fields" not in st.session_state:
        st.session_state.fields = ["text", "intent", "sentiment", "sentiment_polarity", "tone", "category", "keywords"]
    if "examples" not in st.session_state:
        st.session_state.examples = [
            '"Hi there!"',
            '"greeting"',
            '"positive"',
            '0.9',
            '"friendly"',
            '"interaction"',
            '"hi" "hello" "welcome"'
        ]

    # Display existing fields and examples with input boxes
    for i in range(len(st.session_state.fields)):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.fields[i] = st.text_input(f"Field {i + 1} Name", value=st.session_state.fields[i], key=f"field_name_{i}")
        with col2:
            st.session_state.examples[i] = st.text_input(f"Example for '{st.session_state.fields[i]}'", value=st.session_state.examples[i], key=f"example_text_{i}")

    # Button to add new field-example pair
    if st.button("➕ Add New Field"):
        st.session_state.fields.append(f"field_{len(st.session_state.fields) + 1}")
        st.session_state.examples.append("")

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

    # Initialize Davyd
    generator = Davyd(num_entries=num_entries, ollama_host=ollama_host, ollama_model=selected_model)

    # Generate Dataset Button
    if st.button("✨ Generate Dataset"):
        with st.spinner("Generating dataset..."):
            try:
                generator.generate_dataset(heading, example_rows)
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
