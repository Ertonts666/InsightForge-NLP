"""
Multilingual Sentiment Analysis module for the NLP Insights Engine.
This module provides functionality to analyze sentiment in multiple languages.
"""

import os
from typing import Dict, List, Optional, Union, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A class for multilingual sentiment analysis using pre-trained transformer models.
    
    Supports multiple languages and provides detailed sentiment analysis with
    confidence scores and fine-grained sentiment categories.
    """
    
    # Mapping of language codes to pre-trained models
    LANGUAGE_MODEL_MAPPING = {
        'en': 'distilbert-base-uncased-finetuned-sst-2-english',
        'multilingual': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'zh': 'uer/roberta-base-finetuned-jd-binary-chinese',
        'fr': 'tblard/tf-allocine',
        'de': 'oliverguhr/german-sentiment-bert',
        'es': 'dccuchile/bert-base-spanish-wwm-uncased-sentiment',
    }
    
    def __init__(self, default_language: str = 'en', device: str = None):
        """
        Initialize the SentimentAnalyzer with specified language and device.
        
        Args:
            default_language: Default language code ('en', 'fr', 'de', 'es', 'zh')
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.default_language = default_language
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing SentimentAnalyzer with default language '{default_language}' on {self.device}")
        
        # Initialize models lazily
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Load default language model
        self._load_model(default_language)
    
    def _load_model(self, language: str) -> None:
        """
        Load the model for a specific language.
        
        Args:
            language: Language code to load the model for
        """
        if language in self.models:
            return
        
        if language not in self.LANGUAGE_MODEL_MAPPING:
            logger.warning(f"Language '{language}' not supported, falling back to multilingual model")
            language = 'multilingual'
        
        model_name = self.LANGUAGE_MODEL_MAPPING[language]
        logger.info(f"Loading sentiment model for {language}: {model_name}")
        
        try:
            self.tokenizers[language] = AutoTokenizer.from_pretrained(model_name)
            self.models[language] = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipelines[language] = pipeline(
                "sentiment-analysis",
                model=self.models[language],
                tokenizer=self.tokenizers[language],
                device=0 if self.device == 'cuda' else -1
            )
            logger.info(f"Successfully loaded model for {language}")
        except Exception as e:
            logger.error(f"Error loading model for {language}: {str(e)}")
            raise
    
    def analyze(self, text: str, language: Optional[str] = None) -> Dict:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: Text to analyze
            language: Language code (defaults to the instance's default language)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not text.strip():
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "language": language or self.default_language
            }
        
        lang = language or self.default_language
        
        # Load model if not already loaded
        if lang not in self.models:
            self._load_model(lang)
        
        # Run sentiment analysis
        try:
            result = self.pipelines[lang](text)
            
            # Process and normalize the result
            processed_result = self._process_result(result, lang)
            processed_result["language"] = lang
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "sentiment": "error",
                "score": 0.0,
                "confidence": 0.0,
                "error": str(e),
                "language": lang
            }
    
    def _process_result(self, result: List[Dict], language: str) -> Dict:
        """
        Process and normalize the model output to a standard format.
        
        Args:
            result: Raw output from the sentiment pipeline
            language: Language code
            
        Returns:
            Normalized sentiment result
        """
        # Different models have different output formats
        if language == 'en':
            # Binary sentiment (positive/negative)
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            sentiment_map = {
                'positive': 1.0,
                'negative': 0.0,
                'neutral': 0.5
            }
            
            return {
                "sentiment": label,
                "score": sentiment_map.get(label, 0.5),
                "confidence": score
            }
            
        elif language == 'multilingual':
            # 5-star rating (1-5)
            label = result[0]['label']
            score = result[0]['score']
            
            # Convert 1-5 star rating to sentiment
            star_rating = int(label.split()[0])
            
            if star_rating >= 4:
                sentiment = "positive"
                normalized_score = 0.75 + (star_rating - 4) * 0.25
            elif star_rating <= 2:
                sentiment = "negative"
                normalized_score = 0.25 * star_rating
            else:
                sentiment = "neutral"
                normalized_score = 0.5
                
            return {
                "sentiment": sentiment,
                "score": normalized_score,
                "confidence": score,
                "raw_rating": star_rating
            }
            
        else:
            # Default processing for other models
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Normalize label names
            if 'pos' in label or 'positive' in label:
                sentiment = 'positive'
                normalized_score = 0.75 + (score * 0.25)
            elif 'neg' in label or 'negative' in label:
                sentiment = 'negative'
                normalized_score = 0.25 - (score * 0.25)
            else:
                sentiment = 'neutral'
                normalized_score = 0.5
                
            return {
                "sentiment": sentiment,
                "score": normalized_score,
                "confidence": score,
                "raw_label": label
            }
    
    def batch_analyze(self, texts: List[str], language: Optional[str] = None) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            language: Language code (defaults to the instance's default language)
            
        Returns:
            List of dictionaries containing sentiment analysis results
        """
        lang = language or self.default_language
        
        # Load model if not already loaded
        if lang not in self.models:
            self._load_model(lang)
        
        results = []
        for text in texts:
            results.append(self.analyze(text, lang))
        
        return results
    
    def analyze_with_aspects(self, text: str, aspects: List[str], language: Optional[str] = None) -> Dict:
        """
        Analyze sentiment with respect to specific aspects mentioned in the text.
        
        Args:
            text: Text to analyze
            aspects: List of aspects to analyze sentiment for
            language: Language code
            
        Returns:
            Dictionary with overall sentiment and per-aspect sentiment
        """
        lang = language or self.default_language
        
        # Get overall sentiment
        overall_sentiment = self.analyze(text, lang)
        
        # Extract sentences or relevant parts for each aspect
        # This is a simplified implementation - a production system would use
        # more sophisticated methods like aspect-based sentiment analysis models
        aspect_results = {}
        
        for aspect in aspects:
            # Find sentences containing the aspect
            sentences = [s for s in text.split('.') if aspect.lower() in s.lower()]
            
            if sentences:
                # Analyze sentiment for these sentences
                aspect_sentiment = self.batch_analyze(sentences, lang)
                
                # Aggregate results
                avg_score = sum(result['score'] for result in aspect_sentiment) / len(aspect_sentiment)
                
                aspect_results[aspect] = {
                    "sentiment": "positive" if avg_score > 0.6 else "negative" if avg_score < 0.4 else "neutral",
                    "score": avg_score,
                    "confidence": sum(result['confidence'] for result in aspect_sentiment) / len(aspect_sentiment),
                    "mentions": len(sentences)
                }
            else:
                aspect_results[aspect] = {
                    "sentiment": "not_mentioned",
                    "score": 0.5,
                    "confidence": 0.0,
                    "mentions": 0
                }
        
        return {
            "overall": overall_sentiment,
            "aspects": aspect_results
        }
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(self.LANGUAGE_MODEL_MAPPING.keys())
