import os
import json
import torch
from typing import Dict, Optional, List
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from functools import lru_cache
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize sentiment analyzer with configurable device.

        Args:
            device: 'cuda', 'cpu', or None for automatic detection
        """
        self.device = self._determine_device(device)
        self.rating_model = None
        self.confusion_model = None
        self._initialize_models()

    def _determine_device(self, device: Optional[str]) -> str:
        """Determine the best available device"""
        if device:
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _initialize_models(self):
        """Initialize models with error handling and progress indication"""
        logger.info("Initializing sentiment models...")

        # Rating model (0-5 stars)
        try:
            self.rating_model = self._load_model(
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "text-classification"
            )
        except Exception as e:
            logger.error(f"Failed to load rating model: {e}")
            self.rating_model = None

        # Confusion detection model
        try:
            self.confusion_model = self._load_model(
                "facebook/bart-large-mnli",
                "zero-shot-classification"
            )
        except Exception as e:
            logger.error(f"Failed to load confusion model: {e}")
            self.confusion_model = None

    @lru_cache(maxsize=1)
    def _load_model(self, model_name: str, task: str):
        """Load and cache a single model"""
        return pipeline(
            task,
            model=model_name,
            device=self.device,
            truncation=True
        )

    def analyze(self, text: str) -> Dict:
        """
        Perform comprehensive sentiment analysis.

        Returns:
            {
                "sentiment": str (VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE),
                "score": float (0-5),
                "confidence": float (0-1),
                "vader_scores": {
                    "neg": float,
                    "neu": float,
                    "pos": float,
                    "compound": float
                },
                "confusion": {
                    "status": "CONFUSED" or "NOT_CONFUSED",
                    "confidence": float (0-1)
                } or None
            }
        """
        result = {
            "sentiment": "NEUTRAL",
            "score": 3.0,
            "confidence": 0.5,
            "vader_scores": {
                "neg": 0.0,
                "neu": 1.0,
                "pos": 0.0,
                "compound": 0.0
            },
            "confusion": None
        }

        try:
            # Get sentiment rating if model is available
            if self.rating_model:
                rating_result = self._safe_predict(self.rating_model, text)
                if rating_result:
                    label = rating_result[0]["label"]
                    score = int(label.split()[0]) if " " in label else int(label)
                    confidence = float(rating_result[0]["score"])

                    # Map score to sentiment categories
                    sentiment_map = {
                        1: "VERY_NEGATIVE",
                        2: "NEGATIVE",
                        3: "NEUTRAL",
                        4: "POSITIVE",
                        5: "VERY_POSITIVE"
                    }

                    result.update({
                        "sentiment": sentiment_map.get(score, "NEUTRAL"),
                        "score": float(max(0, min(5, score))),
                        "confidence": confidence
                    })

            # Detect confusion if model is available
            if self.confusion_model:
                confusion_result = self._safe_predict(
                    self.confusion_model,
                    text,
                    candidate_labels=["confused", "not confused"],
                    multi_label=False
                )

                if confusion_result:
                    is_confused = (confusion_result["labels"][0] == "confused" and
                                   confusion_result["scores"][0] > 0.6)
                    result["confusion"] = {
                        "status": "CONFUSED" if is_confused else "NOT_CONFUSED",
                        "confidence": float(confusion_result["scores"][0])
                    }

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            result["error"] = str(e)

        return result

    def _safe_predict(self, model, text: str, **kwargs):
        """Safe prediction with length handling and error catching"""
        try:
            if len(text) > 512:
                text = text[:512]
                logger.warning("Text truncated to 512 characters for model input")
            return model(text, **kwargs)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts efficiently"""
        return [self.analyze(text) for text in tqdm(texts, desc="Analyzing texts")]