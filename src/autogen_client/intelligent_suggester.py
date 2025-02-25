import logging
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from ollama_client import OllamaClient  # Ensure this import is correct


class IntelligentSuggester:
    """
    A class that provides intelligent suggestions for field names, types, and values
    based on user input or existing data.
    """

    def __init__(self, ollama_client: OllamaClient):
        self.field_suggestions = [
            "text", "intent", "sentiment", "tone", "category", "keywords", "polarity", "confidence"
        ]
        self.example_data = [
            "This is a positive example",
            "Request to draw a mountain landscape",
            "Feeling sad about recent events",
            "Congratulations on your achievement"
        ]
        self.vectorizer = TfidfVectorizer()
        self.client = ollama_client  # Store the OllamaClient instance

        logging.info("IntelligentSuggester initialized with default field suggestions.")

    def suggest_field_names(self, user_input: str) -> List[str]:
        """
        Suggest field names based on user input.
        
        :param user_input: A string representing the user's input or description.
        :return: A list of suggested field names.
        """
        suggestions = []
        
        # Optionally use OllamaClient to generate suggestions
        prompt = f"Based on the following description, suggest relevant field names: {user_input}"
        try:
            response = self.client.generate_text(prompt)
            # Assuming the response is a comma-separated string of field names
            suggestions = [s.strip() for s in response.split(',')]
            logging.info(f"Field name suggestions for '{user_input}': {suggestions}")
        except Exception as e:
            logging.error(f"Error in suggesting field names using OllamaClient: {e}")
            # Fallback to existing field_suggestions based on user_input
            for suggestion in self.field_suggestions:
                if user_input.lower() in suggestion.lower():
                    suggestions.append(suggestion)
            logging.info(f"Fallback field name suggestions for '{user_input}': {suggestions}")
        
        return suggestions

    def suggest_field_values(self, data: List[str], target_field: str) -> List[str]:
        """
        Suggest values for a specific field based on existing data.
        
        :param data: A list of strings representing the dataset.
        :param target_field: The name of the field for which values are to be suggested.
        :return: A list of suggested values.
        """
        if not data:
            logging.warning("No data provided for field value suggestion.")
            return []

        # Fit vectorizer to example data and transform input data
        self.vectorizer.fit(self.example_data)
        data_vectors = self.vectorizer.transform(data)

        # Compute similarity with example data
        example_vectors = self.vectorizer.transform(self.example_data)
        similarities = cosine_similarity(data_vectors, example_vectors)

        # Find the most similar examples
        suggestions = []
        for i, row in enumerate(similarities):
            max_index = row.argmax()
            suggestions.append(self.example_data[max_index])
            logging.info(f"Suggestion for row {i} in field '{target_field}': {self.example_data[max_index]}")

        return suggestions
    
    def suggest_field_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Suggest data types for each field in the dataset.
        
        :param data: A pandas DataFrame representing the dataset.
        :return: A dictionary mapping field names to suggested data types.
        """
        type_suggestions = {}
        for column in data.columns:
            dtype = data[column].dtype
            if pd.api.types.is_string_dtype(dtype):
                type_suggestions[column] = "string"
            elif pd.api.types.is_numeric_dtype(dtype):
                type_suggestions[column] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_suggestions[column] = "datetime"
            else:
                type_suggestions[column] = "unknown"
            logging.info(f"Suggested type for field '{column}': {type_suggestions[column]}")
        return type_suggestions
