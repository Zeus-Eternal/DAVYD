# src/prompt_engineering.py
PROVIDER_PROMPT_TEMPLATES = {
    "Ollama": "Generate pipe-separated data following this structure:",
    "Gemini": "You are a data generation expert. Create CSV-like data with:",
    "DeepSeek": "[INST] Generate dataset with fields: {fields} [/INST]",
    "ChatGPT": "As a data generation assistant, create records containing:"
}

def get_provider_prompt(provider: str, fields: list) -> str:
    return PROVIDER_PROMPT_TEMPLATES.get(provider, "").format(fields="|".join(fields))