# src/ollama_client.py

import logging
from ollama import Client  # Ensure this import aligns with the actual library
import os
import json
import logging

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')

def validate_json_files():
    for file in os.listdir(TEMPLATE_DIR):
        if file.endswith('.json'):
            filepath = os.path.join(TEMPLATE_DIR, file)
            try:
                with open(filepath, 'r') as f:
                    json.load(f)
                logging.info(f"Template '{file}' is valid.")
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in '{file}': {e}")
            except Exception as e:
                logging.error(f"Error reading '{file}': {e}")

if __name__ == "__main__":
    validate_json_files()

class OllamaClient:
    def __init__(self, model: str = "llama3.2:latest", host: str = "http://localhost:11434", timeout: int = 60):
        """
        Initialize the OllamaClient using the Ollama Python library.

        :param model: The Ollama model to use for text generation.
        :param host: The host URL for the Ollama API.
        :param timeout: Timeout for API requests in seconds.
        """
        self.model = model
        self.host = host
        self.client = Client(host=host, timeout=timeout)
        logging.info(f"OllamaClient initialized with model '{self.model}' and host '{self.host}'.")

    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Ollama model with specified parameters.

        :param prompt: The prompt to send to the model.
        :return: Generated text from Ollama.
        """
        try:
            logging.debug(f"Generating text with prompt: {prompt}")
            
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
                # Removed temperature and max_tokens
            )
            
            generated_text = response['message']['content'].strip()
            logging.info("Text generation successful.")
            return generated_text
        except TypeError as te:
            logging.error(f"TypeError during text generation: {te}")
            raise
        except ValueError as ve:
            logging.error(f"ValueError during text generation: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error generating text with Ollama: {e}")
            raise

    def list_models(self) -> list:
        """
        Fetch the list of available models from the Ollama API.

        :return: List of available model names.
        """
        try:
            logging.debug("Fetching available models from Ollama API.")
            models_response = self.client.list()
            
            # Handle different possible response structures
            if isinstance(models_response, list):
                # Assume each item is a dict with a 'name' key
                model_names = [model['name'] for model in models_response if 'name' in model]
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Assume 'models' key contains the list
                model_names = [model['name'] for model in models_response['models'] if 'name' in model]
            else:
                # Unexpected format
                logging.warning("Unexpected format for models response.")
                model_names = []
            
            logging.info(f"Fetched {len(model_names)} models from Ollama API.")
            return model_names
        except Exception as e:
            logging.error(f"Failed to fetch models: {e}")
            return []

    def pull(self, model: str):
        """
        Pull a model from the Ollama repository.

        :param model: The model name to pull.
        """
        try:
            logging.info(f"Pulling model '{model}' from Ollama.")
            self.client.pull(model)
            logging.info(f"Model '{model}' pulled successfully.")
        except Exception as e:
            logging.error(f"Failed to pull model '{model}': {e}")
            raise

    def delete(self, model: str):
        """
        Delete a model from the Ollama repository.

        :param model: The model name to delete.
        """
        try:
            logging.info(f"Deleting model '{model}' from Ollama.")
            self.client.delete(model)
            logging.info(f"Model '{model}' deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete model '{model}': {e}")
            raise
