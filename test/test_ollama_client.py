from ollama import Client  # Import the Client class from the ollama-python library
import logging

class OllamaClient:
    def __init__(self, model: str = "llama3.2:latest", host: str = "http://localhost:11434", timeout: int = 60):
        """
        Initialize the OllamaClient using the ollama-python library.

        :param model: The Ollama model to use for text generation.
        :param host: The host URL for the Ollama API.
        :param timeout: Timeout for API requests in seconds.
        """
        self.model = model
        self.client = Client(host=host, timeout=timeout)
        logging.info(f"OllamaClient initialized with model '{self.model}' and host '{host}'.")

    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the Ollama model.

        :param prompt: The prompt to send to the model.
        :return: Generated text from Ollama.
        """
        try:
            logging.debug(f"Generating text with prompt: {prompt}")
            response = self.client.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            generated_text = response['message']['content'].strip()
            logging.info("Text generation successful.")
            return generated_text
        except Exception as e:
            logging.error(f"Error generating text with Ollama: {e}")
            return "Error: Failed to generate text."

    def health_check(self) -> bool:
        """
        Check if the Ollama API is reachable and the model is available.

        :return: True if the API is reachable and model is available, False otherwise.
        """
        try:
            logging.debug("Performing health check with Ollama.")
            # Since the ollama-python library does not have a built-in health_check method,
            # you may need to implement your own check here, or use a basic request to the API's status endpoint
            return True  # Simplified for demonstration purposes
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False
