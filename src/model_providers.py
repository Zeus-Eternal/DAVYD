# src/model_providers.py
import os
import logging
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
import openai
import anthropic
from ollama import Client  # Ensure this import aligns with the actual library
import requests  # For Mistral, Groq, and Hugging Face APIs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModelClient:
    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError
        
    def list_models(self) -> List[str]:
        raise NotImplementedError

class OllamaClient(BaseModelClient):
    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434", timeout: int = 60):
        """
        Initialize the OllamaClient using the Ollama Python library.

        :param model: The Ollama model to use for text generation.
        :param host: The host URL for the Ollama API.
        :param timeout: Timeout for API requests in seconds.
        """
        self.model = model
        self.host = host
        self.client = Client(host=host, timeout=timeout)
        logger.info(f"OllamaClient initialized with model '{self.model}' and host '{self.host}'.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Ollama model with specified parameters.

        :param prompt: The prompt to send to the model.
        :return: Generated text from Ollama.
        """
        try:
            logger.debug(f"Generating text with prompt: {prompt}")
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            generated_text = response['message']['content'].strip()
            logger.info("Text generation successful.")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            raise

    def list_models(self) -> List[str]:
        """
        Fetch the list of available models from the Ollama API.

        :return: List of available model names.
        """
        try:
            logger.debug("Fetching available models from Ollama API.")
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
                logger.warning("Unexpected format for models response.")
                model_names = []
            
            logger.info(f"Fetched {len(model_names)} models from Ollama API.")
            return model_names
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return ["llama2"]  # Default fallback model

    def pull(self, model: str):
        """
        Pull a model from the Ollama repository.

        :param model: The model name to pull.
        """
        try:
            logger.info(f"Pulling model '{model}' from Ollama.")
            self.client.pull(model)
            logger.info(f"Model '{model}' pulled successfully.")
        except Exception as e:
            logger.error(f"Failed to pull model '{model}': {e}")
            raise

    def delete(self, model: str):
        """
        Delete a model from the Ollama repository.

        :param model: The model name to delete.
        """
        try:
            logger.info(f"Deleting model '{model}' from Ollama.")
            self.client.delete(model)
            logger.info(f"Model '{model}' deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete model '{model}': {e}")
            raise

class DeepSeekClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error generating text with DeepSeek: {e}")
            raise

    def list_models(self) -> List[str]:
        return ["deepseek-chat"]

class GeminiClient(BaseModelClient):
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.error(f"Failed to configure Gemini client: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            
            # Check if the response is valid
            if not hasattr(response, 'text') or not response.text:
                logger.error("Gemini API returned an invalid response.")
                logger.error(f"Response: {response}")
                raise ValueError("Invalid response from Gemini API")
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            logger.error(f"Prompt: {prompt}")  # Log the prompt for debugging
            raise

    def list_models(self) -> List[str]:
        try:
            # Gemini currently supports only one model
            return ["gemini-pro"]
        except Exception as e:
            logger.error(f"Failed to list Gemini models: {e}")
            return []
        
class ChatGPTClient(BaseModelClient):
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with ChatGPT: {e}")
            raise

    def list_models(self) -> List[str]:
        return ["gpt-4", "gpt-3.5-turbo"]

class AnthropicClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key=api_key)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.completion(
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-2",
                max_tokens_to_sample=1000,
            )
            return response['completion']
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            raise

    def list_models(self) -> List[str]:
        return ["claude-2", "claude-instant-1"]

class ClaudeClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key=api_key)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.completion(
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-2",
                max_tokens_to_sample=1000,
            )
            return response['completion']
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            raise

    def list_models(self) -> List[str]:
        return ["claude-2", "claude-instant-1"]

class MistralClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "mistral-medium",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error generating text with Mistral: {e}")
            raise

    def list_models(self) -> List[str]:
        return ["mistral-medium", "mistral-small"]

class GroqClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/v1"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "groq-1",
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error generating text with Groq: {e}")
            raise

    def list_models(self) -> List[str]:
        return ["groq-1", "groq-2"]

class HuggingFaceClient(BaseModelClient):
    def __init__(self, api_key: str, endpoint: Optional[str] = None):
        """
        Initialize the HuggingFaceClient.

        :param api_key: Hugging Face API key.
        :param endpoint: Custom endpoint URL (optional). If not provided, uses Hugging Face's hosted API.
        """
        self.api_key = api_key
        self.endpoint = endpoint or "https://api-inference.huggingface.co/models"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt
            }
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()[0]['generated_text']
        except Exception as e:
            logger.error(f"Error generating text with Hugging Face: {e}")
            raise

    def list_models(self) -> List[str]:
        # Hugging Face does not provide a direct API to list models, so return a default list
        return ["gpt2", "flan-t5-large", "mistral-7b"]

# Factory function for creating clients
def get_model_client(provider: str, **kwargs) -> BaseModelClient:
    providers = {
        "ollama": OllamaClient,
        "deepseek": DeepSeekClient,
        "gemini": GeminiClient,
        "chatgpt": ChatGPTClient,
        "anthropic": AnthropicClient,
        "claude": ClaudeClient,
        "mistral": MistralClient,
        "groq": GroqClient,
        "huggingface": HuggingFaceClient
    }
    
    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")
    
    return provider_class(**kwargs)