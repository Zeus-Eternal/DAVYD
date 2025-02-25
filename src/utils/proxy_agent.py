# src/utils/proxy_agent.py
from typing import List, Dict, Any
import logging
from autogen_client.assistant_agent import AssistantAgent

class ProxyAgent:
    def __init__(self, config: Dict[str, Any]):
        """Initialize a Proxy Agent to manage multiple Assistant Agents.
        
        Args:
            config: Configuration dictionary with:
                - model_client: The AI model client to use
                - model_name: Name of the model
                - max_retries: Number of retry attempts
                - num_agents: Number of Assistant Agents to manage
        """
        self.config = config
        self.agents = self._initialize_agents()
        self.logger = logging.getLogger(__name__)

    def _initialize_agents(self) -> List[AssistantAgent]:
        """Initialize and return a list of Assistant Agents."""
        return [
            AssistantAgent(name=f"AssistantAgent-{i}", config=self.config)
            for i in range(self.config.get("num_agents", 1))
        ]

    def distribute_task(self, prompt: str) -> List[str]:
        """Distribute a task across all Assistant Agents and collect results."""
        results = []
        for agent in self.agents:
            try:
                result = agent.generate_dataset_entry(prompt)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Agent {agent.name} failed: {str(e)}")
        return results

    def validate_results(self, results: List[str]) -> Dict[str, List[str]]:
        """Validate results from Assistant Agents and categorize them."""
        valid_results = []
        invalid_results = []
        
        for result in results:
            if self.agents[0].validate_entry(result):  # Use first agent's validation
                valid_results.append(result)
            else:
                invalid_results.append(result)
        
        return {
            "valid": valid_results,
            "invalid": invalid_results
        }

    def handle_retries(self, invalid_results: List[str]) -> List[str]:
        """Handle retries for invalid results using all available agents."""
        successful_retries = []
        for agent in self.agents:
            retry_results = agent.retry_failed_entries(invalid_results)
            successful_retries.extend(retry_results)
            invalid_results = [res for res in invalid_results if res not in retry_results]
            if not invalid_results:
                break
        return successful_retries

    def run_pipeline(self, prompt: str) -> Dict[str, List[str]]:
        """Run the full pipeline: generation, validation, and retries."""
        # Step 1: Generate initial results
        initial_results = self.distribute_task(prompt)
        
        # Step 2: Validate results
        validation_results = self.validate_results(initial_results)
        
        # Step 3: Retry invalid results
        if validation_results["invalid"]:
            retry_results = self.handle_retries(validation_results["invalid"])
            validation_results["valid"].extend(retry_results)
        
        return validation_results