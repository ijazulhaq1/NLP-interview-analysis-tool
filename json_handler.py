import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JsonHandler:
    @staticmethod
    def create_topic_modeling_input(source_data: Dict, output_path: str) -> bool:
        """
        Creates topic_modeling_input.json from fully processed data
        Structure:
        {
            "documents": [
                {
                    "text": "processed text",
                    "metadata": {
                        "interview_id": "n01",
                        "question_id": "1.1"
                    }
                },
                ...
            ],
            "stats": {
                "total_documents": int,
                "generated_at": "timestamp"
            }
        }
        """
        try:
            documents = []
            for interview_id, questions in source_data.items():
                for question_id, text in questions.items():
                    if text.strip():  # Only include non-empty texts
                        documents.append({
                            "text": text,
                            "metadata": {
                                "interview_id": interview_id,
                                "question_id": question_id
                            }
                        })

            output = {
                "documents": documents,
                "stats": {
                    "total_documents": len(documents),
                    "generated_at": datetime.now().isoformat()
                }
            }

            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, "topic_modeling_input.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)

            logger.info(f"Created topic modeling input with {len(documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to create topic modeling input: {str(e)}")
            return False

    @staticmethod
    def load_topic_modeling_input(file_path: str) -> List[str]:
        """Load just the text documents for BERTopic from topic_modeling_input.json"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [doc["text"] for doc in data.get("documents", []) if doc.get("text")]
        except Exception as e:
            logger.error(f"Error loading topic modeling input: {str(e)}")
            return []

    @staticmethod
    def group_questions(json_dict: Dict) -> Dict:
        """
        Group questions from multiple interviews (legacy method)
        Returns: {"question1": "combined text", "question2": "combined text"}
        """
        question_dict = {}
        for interview, questions in json_dict.items():
            for question, text in questions.items():
                if question in question_dict:
                    question_dict[question] += " " + text
                else:
                    question_dict[question] = text
        return question_dict

    @staticmethod
    def create_json(json_dict: Dict, filename: str, output_path: str) -> bool:
        """Generic JSON file creation with error handling"""
        try:
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, filename), 'w', encoding='utf-8') as f:
                json.dump(json_dict, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            logger.error(f"Failed to create {filename}: {str(e)}")
            return False

    @staticmethod
    def read_json(file_path: str) -> Dict:
        """Generic JSON file reading with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {file_path}")
            return {}
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return {}

    @staticmethod
    def create_topic_results(topic_data: Dict, output_path: str) -> bool:
        """
        Save topic modeling results with standardized naming
        Args:
            topic_data: Dictionary containing topic results
            output_path: Directory to save the file
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(output_path, exist_ok=True)

            # Get algorithm from topic_data if available
            algorithm = topic_data.get("algorithm", "unknown")

            # Determine filename based on scope and group
            group_name = topic_data.get("group", "")
            if "block_" in group_name:
                filename = f"topic_results_{algorithm}_block_{group_name.split('_')[-1]}.json"
            elif "q_" in group_name:
                filename = f"topic_results_{algorithm}_q_{group_name.split('_')[-1]}.json"
            else:
                filename = f"topic_results_{algorithm}_all_interviews.json"

            filepath = os.path.join(output_path, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(topic_data, f, ensure_ascii=False, indent=4)

            logger.info(f"Saved topic results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save topic results: {str(e)}")
            return False

    @staticmethod
    def create_sentiment_results(sentiment_data: Dict, output_path: str) -> bool:
        """
        Save sentiment analysis results with standardized naming
        Args:
            sentiment_data: Dictionary containing sentiment results
            output_path: Directory to save the file
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(output_path, exist_ok=True)

            # Determine group name and scope
            group_name = sentiment_data.get("group", "")
            if not group_name and "overall_sentiment" in sentiment_data:
                group_name = sentiment_data["overall_sentiment"].get("group", "")

            # Determine filename based on scope
            if "block_" in group_name:
                filename = f"sentiment_results_block_{group_name.split('_')[-1]}.json"
            elif "q_" in group_name:
                filename = f"sentiment_results_q_{group_name.split('_')[-1]}.json"
            else:
                filename = "sentiment_results_all_interviews.json"

            filepath = os.path.join(output_path, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sentiment_data, f, ensure_ascii=False, indent=4)

            logger.info(f"Saved sentiment results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save sentiment results: {str(e)}")
            return False

    @staticmethod
    def parse_questions(content: str) -> Dict:
        """
        Parse questions from raw text (legacy method)
        Format: "-.Pregunta X.-" followed by text
        """
        try:
            questions = re.split(r'-\.Pregunta ([\d\.a-zA-Z]+)\.-', content)[1:]
            parsed_data = {}
            for i in range(0, len(questions), 2):
                question_number = questions[i].strip()
                question_text = re.sub(
                    r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+',
                    '',
                    questions[i + 1].strip()
                )
                parsed_data[question_number] = question_text.replace('\n', ' ').replace('  ', ' ')
            return parsed_data
        except Exception as e:
            logger.error(f"Question parsing failed: {str(e)}")
            return {}