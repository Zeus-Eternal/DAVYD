import unittest
import os, sys
from unittest.mock import patch, MagicMock

# Dynamically set the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.davyd import Davyd

class TestDavydDatasetGenerator(unittest.TestCase):

    def setUp(self):
        """
        Set up a Davyd instance and initial parameters for testing.
        """
        self.davyd = Davyd(
            num_entries=5,
            ollama_url="http://127.0.0.1:11434",
            temperature=0.5,
            max_tokens=150
        )
        self.heading = '"text"|"intent"|"sentiment"|"sentiment_polarity"|"tone"|"category"|"keywords"'
        self.body_examples = [
            '"Hi there!"|greeting|positive|0.9|friendly|interaction|"hi" "hello" "welcome"',
            '"Please draw a mountain landscape."|draw_request|neutral|0.5|instructive|art|"draw" "landscape" "mountain"',
            '"I am feeling sad today."|emotion|negative|0.7|somber|personal|"sad" "unhappy" "down"',
            '"Can you provide the latest sales report?"|information_request|neutral|0.6|business|"sales" "report" "data"',
            '"Congratulations on your promotion!"|celebration|positive|0.95|joyful|work|"congratulations" "promotion" "achievement"'
        ]

    @patch('src.ollama_client.Client.chat')
    def test_generate_dataset_valid_heading(self, mock_chat):
        """
        Test generating a dataset with a valid heading and example body rows.
        """
        mock_response = {'message': {'content': '"Hello!"|greeting|positive|0.8|friendly|interaction|"hello" "hi"'}}
        mock_chat.return_value = mock_response

        self.davyd.generate_dataset(self.heading, self.body_examples)
        self.assertEqual(len(self.davyd.dataset), 5)
        self.assertIsInstance(self.davyd.dataset, list)
        self.assertIsInstance(self.davyd.dataset[0], dict)
        self.assertIn("text", self.davyd.dataset[0])
        self.assertIn("intent", self.davyd.dataset[0])
        self.assertEqual(self.davyd.dataset[0]["text"], "Hello!")
        self.assertEqual(self.davyd.dataset[0]["intent"], "greeting")

    @patch('src.ollama_client.Client.chat')
    def test_generate_dataset_invalid_heading(self, mock_chat):
        """
        Test that an invalid heading raises a ValueError.
        """
        invalid_heading = '"text"|"intent"|"sentiment"|"sentiment_polarity|tone"|"category"|"keywords"'  # Missing closing quote
        with self.assertRaises(ValueError):
            self.davyd.generate_dataset(invalid_heading, self.body_examples)

    def test_parse_example(self):
        """
        Test parsing an example row into a dictionary.
        """
        parsed_example = self.davyd.parse_example(self.body_examples[0], self.heading.split("|"))
        expected = {
            "text": "Hi there!",
            "intent": "greeting",
            "sentiment": "positive",
            "sentiment_polarity": "0.9",
            "tone": "friendly",
            "category": "interaction",
            "keywords": 'hi" "hello" "welcome"'
        }
        self.assertEqual(parsed_example, expected)

    def test_add_entry(self):
        """
        Test adding a new entry to the dataset.
        """
        new_entry = {
            "text": "Hello!",
            "intent": "greeting",
            "sentiment": "positive",
            "sentiment_polarity": "0.8",
            "tone": "friendly",
            "category": "interaction",
            "keywords": "hello|hi|greet"
        }
        self.davyd.add_entry(new_entry)
        self.assertIn(new_entry, self.davyd.dataset)

    def test_modify_entry_valid_index(self):
        """
        Test modifying an existing entry at a valid index.
        """
        self.davyd.generate_dataset(self.heading, self.body_examples)
        updated_entry = {
            "text": "Goodbye!",
            "intent": "farewell",
            "sentiment": "neutral",
            "sentiment_polarity": "0.5",
            "tone": "neutral",
            "category": "interaction",
            "keywords": "bye|farewell|goodbye"
        }
        self.davyd.modify_entry(0, updated_entry)
        self.assertEqual(self.davyd.dataset[0], updated_entry)

    def test_modify_entry_invalid_index(self):
        """
        Test that modifying an entry with an invalid index raises an IndexError.
        """
        with self.assertRaises(IndexError):
            self.davyd.modify_entry(10, {})

    def test_delete_entry_valid_index(self):
        """
        Test deleting an entry at a valid index.
        """
        self.davyd.generate_dataset(self.heading, self.body_examples)
        initial_length = len(self.davyd.dataset)
        self.davyd.delete_entry(0)
        self.assertEqual(len(self.davyd.dataset), initial_length - 1)

    def test_delete_entry_invalid_index(self):
        """
        Test that deleting an entry with an invalid index raises an IndexError.
        """
        with self.assertRaises(IndexError):
            self.davyd.delete_entry(10)

    @patch('src.utils.convert_dict_to_df')
    def test_get_dataset_as_df(self, mock_convert):
        """
        Test converting the dataset to a pandas DataFrame.
        """
        self.davyd.generate_dataset(self.heading, self.body_examples)
        mock_df = MagicMock()
        mock_convert.return_value = mock_df
        df = self.davyd.get_dataset_as_df()
        self.assertEqual(df, mock_df)
        mock_convert.assert_called_once_with(self.davyd.dataset)

if __name__ == "__main__":
    unittest.main()
