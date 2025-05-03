import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AssistantAgent:
    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initialize an Assistant Agent for dataset generation tasks.

        Args:
            config: Configuration dictionary with:
                - model_client: The AI model client implementing generate_text()
                - model_name: Name of the model (optional for tracking/logging)
                - max_retries: Optional retry count for fallback logic
            name: Unique identifier for the agent
        """
        self.name = name
        self.config = config
        self.history: List[str] = []

    def generate_dataset_entry(self, prompt: str) -> Optional[str]:
        """
        Generate a dataset entry using the configured model client.
        
        Args:
            prompt: The input prompt for generation
            
        Returns:
            Generated text as string, or None if generation failed
        """
        try:
            raw_response = self.config["model_client"].generate_text(prompt)
            if not raw_response:
                return None

            # Handle dict responses if model client doesn't return plain string
            if isinstance(raw_response, dict):
                content = (
                    raw_response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                response_str = str(content).strip()
            else:
                response_str = str(raw_response).strip()

            if response_str:
                self.history.append(response_str)
                return response_str

            return None

        except Exception as e:
            logger.error(f"Generation error in assistant: {str(e)}")
            return None

    def validate_entry(self, entry: str) -> bool:
        """
        Validate a generated dataset entry.

        Args:
            entry: The entry to validate

        Returns:
            Boolean indicating whether entry is valid
        """
        return bool(entry and isinstance(entry, str) and entry.strip())

    def retry_failed_entries(self, failed_entries: List[str]) -> List[str]:
        """
        Retry generation for failed entries.

        Args:
            failed_entries: List of prompts that previously failed

        Returns:
            List of successfully generated entries
        """
        successful_retries = []
        for prompt in failed_entries:
            retry_response = self.generate_dataset_entry(prompt)
            if self.validate_entry(retry_response):
                successful_retries.append(retry_response)
        return successful_retries

    def last_message(self) -> Optional[str]:
        """Get the last message from history as a string, never a dict or None"""
        if not self.history:
            return ""
        return self.history[-1]
