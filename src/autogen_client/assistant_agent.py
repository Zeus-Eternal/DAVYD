# src/autogen_client/assistant_agent.py
from typing import List
from typing import Dict, Any

class AssistantAgent:
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize an Assistant Agent for dataset generation tasks.
        
        Args:
            name: Unique identifier for the agent
            config: Configuration dictionary with:
                - model_client: The AI model client to use
                - model_name: Name of the model
                - max_retries: Number of retry attempts
        """
        self.name = name
        self.config = config
        self.history = []
        
    def generate_dataset_entry(self, prompt: str) -> str:
        """Generate a dataset entry using the configured AI model."""
        try:
            response = self.config["model_client"].generate(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return None

    def validate_entry(self, entry: str) -> bool:
        """Validate a generated dataset entry."""
        # Add your validation logic here
        return True  # Simplified for example

    def retry_failed_entries(self, failed_entries: List[str]) -> List[str]:
        """Retry generation for failed entries."""
        successful_retries = []
        for entry in failed_entries:
            retry_response = self.generate_dataset_entry(entry)
            if retry_response and self.validate_entry(retry_response):
                successful_retries.append(retry_response)
        return successful_retries