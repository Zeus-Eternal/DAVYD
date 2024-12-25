# DAVYD: Dynamic AI Virtual Yielding Dataset

![DAVYD Logo](DAVYD_SM.jpg)

**Developer:** agustealo  
**Website:** [agustealo.com](https://agustealo.com)  
**Email:** [agustealo@gmail.com](mailto:agustealo@gmail.com)

---

## Table of Contents

- [What is DAVYD?](#what-is-davyd)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
- [Getting Started](#getting-started)
  - [Run the Application](#run-the-application)
  - [Define Your Dataset](#define-your-dataset)
  - [Export Your Dataset](#export-your-dataset)
- [Acronym Breakdown: DAVYD](#acronym-breakdown-davyd)
- [Features Overview](#features-overview)
  - [1. Streamlit UI](#1-streamlit-ui)
  - [2. AI Model Integration](#2-ai-model-integration)
  - [3. Data Validation](#3-data-validation)
  - [4. Scheduled Generation](#4-scheduled-generation)
- [Usage Examples](#usage-examples)
- [Contribution](#contribution)
  - [How to Contribute](#how-to-contribute)
- [License](#license)
- [Support](#support)

---

## What is DAVYD?
![DAVYD Logo](DAVYD_SM.jpg)

**DAVYD** (Dynamic AI Virtual Yielding Dataset) is an intelligent dataset generator powered by advanced AI models. It allows developers, researchers, and data scientists to generate structured datasets tailored for machine learning and AI workflows. Designed with flexibility and scalability in mind, DAVYD simplifies the process of creating realistic, high-quality datasets that adhere to specific fields and descriptions.

---

## Key Features

- **Customizable Dataset Structure**: Define your own fields and examples to generate structured datasets.
- **AI-Driven Generation**: Leverages cutting-edge AI models, such as `llama3.2:latest` and `deepseek-coder-v2`, for data generation.
- **Validation & Quality Assurance**: Built-in data validation ensures all generated datasets meet specified requirements.
- **Flexible Output Formats**: Save datasets as CSV or JSON for seamless integration with existing workflows.
- **Streamlit Interface**: A user-friendly web interface for defining, generating, and managing datasets.

---

## Installation

### Prerequisites

- **Python 3.9+**
- **Streamlit**
- **Ollama Client** (for AI model integration)

### Clone the Repository

```bash
git clone https://github.com/agustealo/DAVYD.git
cd DAVYD
```

### Install Dependencies

Create a virtual environment and install the required libraries:

```bash
python -m venv env
# Activate the virtual environment:
# On macOS/Linux:
source env/bin/activate
# On Windows:
env\Scripts\activate

pip install -r requirements.txt
```

---

## Getting Started

### Run the Application

Launch the Streamlit interface:

```bash
streamlit run src/ui.py
```

Access the app in your browser at [http://localhost:8501](http://localhost:8501).

### Define Your Dataset

1. **Navigate to the "Define Dataset Structure" Section**:
   - Add fields and examples manually or use a preloaded template.
2. **Generate Datasets**:
   - Click on the "✨ Generate Dataset" button to create your dataset based on the defined structure.

### Export Your Dataset

Once the dataset is generated, export it in CSV or JSON format. You can also preview and edit the data before saving to ensure it meets your requirements.

---

## Acronym Breakdown: DAVYD

- **D**: Dynamic  
- **A**: AI  
- **V**: Virtual  
- **Y**: Yielding  
- **D**: Dataset

---

## Features Overview

### 1. Streamlit UI

- **Interactive Layout**: Define fields, view live previews, and generate datasets with ease.
- **Template Management**: Load, modify, and export dataset templates.
- **Visualization**: View data quality metrics and insights.

### 2. AI Model Integration

- **Multiple AI Models Support**: Integrates with various AI models via the Ollama API.
- **Dynamic Model Fetching**: Automatically fetches and utilizes available models for data generation.

### 3. Data Validation

- **Field Validation**: Ensures all required fields are present and correctly formatted.
- **Consistency Checks**: Validates data types and value ranges to maintain dataset integrity.
- **Detailed Logging**: Provides warnings and errors for any validation issues encountered.

### 4. Scheduled Generation

- **Automated Tasks**: Schedule dataset generation at defined intervals (Hourly, Daily).
- **Task Management**: Manage and monitor scheduled tasks through the interface.

---

## Usage Examples

### Example 1: Generating a Sentiment Analysis Dataset

1. **Define Fields**: `text`, `intent`, `sentiment`, `sentiment_polarity`, `tone`, `category`, `keywords`.
2. **Provide Examples**:
   - `"I love this product!"`, `"affirmation"`, `"positive"`, `0.9`, `"enthusiastic"`, `"review"`, `"love, product"`
   - `"This is disappointing."`, `"complaint"`, `"negative"`, `-0.7`, `"dissatisfied"`, `"experience"`, `"disappointment, bad"`
3. **Generate Dataset**: Click "✨ Generate Dataset" to create 150 entries.
4. **Export**: Save the dataset as `sentiment_analysis.csv`.

### Example 2: Creating an Intent Classification Dataset

1. **Load Template**: Select `intent_classification.json` from the template dropdown.
2. **Review Fields and Examples**.
3. **Generate Dataset**: Click "✨ Generate Dataset" to create 150 entries.
4. **Validate and Export**: Ensure data quality and export as JSON.

---

## Contribution

Contributions are welcome! Feel free to open an issue or submit a pull request to enhance DAVYD.

### How to Contribute

1. **Fork the Repository**:
   - Click the "Fork" button at the top right of the repository page.
2. **Create a New Branch** for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. **Commit Your Changes**:
   ```bash
   git commit -m "Add a feature or fix a bug"
   ```
4. **Push to Your Branch**:
   ```bash
   git push origin feature-name
   ```
5. **Open a Pull Request**:
   - Navigate to your forked repository on GitHub.
   - Click the "Compare & pull request" button.
   - Provide a clear description of your changes and submit the pull request.

---

## License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

---

## Support

If you encounter any issues or have questions, feel free to contact the developer:

- **Email**: [agustealo@gmail.com](mailto:agustealo@gmail.com)
- **Website**: [agustealo.com](https://agustealo.com)

![Agustealo.com](https://agustealo.com/wp-content/uploads/2024/06/agustealo-hztl-logo-BLK-w400.png)
---

## Acknowledgements

- **Streamlit**: For providing a powerful and easy-to-use framework for building interactive web applications.
- **Ollama**: For their AI models that power DAVYD's data generation capabilities.
- **OpenAI**: For inspiring the integration of advanced AI in data generation tools.

---

**Happy dataset generation with DAVYD! 🚀🔥**

---