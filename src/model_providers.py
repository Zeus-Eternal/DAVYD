#!/usr/bin/env python3
# src/model_providers.py

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

import requests
from requests import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional imports for hosted/cloud providers
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from ollama import Client as OllamaAPIClient
except ImportError:
    OllamaAPIClient = None

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


class BaseModelClient:
    """
    Abstract base class for all model clients.
    """

    def generate_text(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        raise NotImplementedError

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        prompt = "\n".join(m["content"] for m in messages)
        text = self.generate_text(prompt, model=model, temperature=temperature, **kwargs)
        return {"choices": [{"message": {"content": text}}]}

    def health_check(self) -> bool:
        try:
            return bool(self.list_models())
        except Exception:
            return False


class OllamaClient(BaseModelClient):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama2"):
        if OllamaAPIClient is None:
            raise ImportError("Ollama Python client not installed.")
        self.client = OllamaAPIClient(host=host)
        self.default_model = model
        self.available_models = self._refresh_models()
        logger.info(f"[Ollama] initialized with {len(self.available_models)} models")

    def _refresh_models(self) -> List[str]:
        try:
            resp = self.client.list()
            if isinstance(resp, list):
                return [m["name"] for m in resp if isinstance(m, dict) and "name" in m]
            elif isinstance(resp, dict) and "models" in resp:
                return [m["name"] for m in resp["models"] if "name" in m]
        except Exception as e:
            logger.warning(f"[Ollama] list_models failed: {e}")
        return [self.default_model]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = self.client.generate(
            model=kwargs.get("model", self.default_model),
            prompt=prompt,
            options={
                "temperature": kwargs.get("temperature", 0.7),
                "top_p":        kwargs.get("top_p", 0.9),
                "num_predict":  kwargs.get("max_tokens", 1024)
            }
        )
        return resp.get("response", "").strip()

    def list_models(self) -> List[str]:
        return self.available_models


class DeepSeekClient(BaseModelClient):
    """
    DeepSeek.com chat completion client.
    """
    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        if not api_key.strip():
            raise ValueError("DeepSeek API key is required")
        self.api_key = api_key.strip()
        self.model_name = model_name
        logger.info("[DeepSeek] initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                json={
                    "model":       self.model_name,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p":       kwargs.get("top_p", 1.0),
                    "max_tokens":  kwargs.get("max_tokens", 2048)
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[DeepSeek] Generation failed: {e}")
            return ""

    def list_models(self) -> List[str]:
        return [self.model_name]

    def health_check(self) -> bool:
        # Always return True; errors surface at generate time
        return True


class GeminiClient(BaseModelClient):
    def __init__(self, api_key: str):
        if genai is None:
            raise ImportError("google.generativeai not installed")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        logger.info("[Gemini] initialized")

    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = self.model.generate_content(prompt)
        text = getattr(resp, "text", "").strip()
        if not text:
            raise ValueError("Empty response from Gemini")
        return text

    def list_models(self) -> List[str]:
        return ["gemini-pro"]


class ChatGPTClient(BaseModelClient):
    def __init__(self, api_key: str):
        if openai is None:
            raise ImportError("openai not installed")
        openai.api_key = api_key
        logger.info("[ChatGPT] initialized")

    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = openai.ChatCompletion.create(
            model=kwargs.get("model", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 0.9)
        )
        return resp.choices[0].message.content.strip()

    def list_models(self) -> List[str]:
        return ["gpt-4", "gpt-3.5-turbo"]


class AnthropicClient(BaseModelClient):
    def __init__(self, api_key: str):
        if anthropic is None:
            raise ImportError("anthropic not installed")
        self.client = anthropic.Client(api_key=api_key)
        logger.info("[Anthropic] initialized")

    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = self.client.completion(
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=kwargs.get("model", "claude-2"),
            max_tokens_to_sample=kwargs.get("max_tokens", 1024)
        )
        return resp["completion"].strip()

    def list_models(self) -> List[str]:
        return ["claude-2", "claude-instant-1"]


class ClaudeClient(AnthropicClient):
    pass


class MistralClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        logger.info("[Mistral] initialized")

    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": kwargs.get("model", "mistral-medium"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7)
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def list_models(self) -> List[str]:
        return ["mistral-medium", "mistral-small"]


class GroqClient(BaseModelClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/v1"
        logger.info("[Groq] initialized")

    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": kwargs.get("model", "groq-1"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7)
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def list_models(self) -> List[str]:
        return ["groq-1", "groq-2"]


class HuggingFaceClient(BaseModelClient):
    def __init__(self, api_key: str, endpoint: Optional[str] = None):
        self.endpoint = endpoint or "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        logger.info("[HuggingFace] initialized")

    def generate_text(self, prompt: str, **kwargs) -> str:
        resp = requests.post(
            self.endpoint,
            headers=self.headers,
            json={"inputs": prompt},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        return data.get("text", "").strip()

    def list_models(self) -> List[str]:
        return ["gpt2", "flan-t5-large", "mistral-7b"]


def get_model_client(provider: str, **kwargs) -> BaseModelClient:
    mapping = {
        "ollama":      OllamaClient,
        "deepseek":    DeepSeekClient,
        "gemini":      GeminiClient,
        "chatgpt":     ChatGPTClient,
        "anthropic":   AnthropicClient,
        "claude":      ClaudeClient,
        "mistral":     MistralClient,
        "groq":        GroqClient,
        "huggingface": HuggingFaceClient,
    }
    cls = mapping.get(provider.lower())
    if not cls:
        raise ValueError(f"Unknown provider: {provider}")
    return cls(**kwargs)
