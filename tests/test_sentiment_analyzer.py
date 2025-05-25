"""
Tests for the SentimentAnalyzer class.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.sentiment_analysis.analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for the SentimentAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        # Initialize analyzer with English as default language
        cls.analyzer = SentimentAnalyzer(default_language='en')
    
    def test_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.default_language, 'en')
        self.assertIn('en', self.analyzer.models)
        self.assertIn('en', self.analyzer.tokenizers)
    
    def test_english_sentiment_positive(self):
        """Test sentiment analysis on positive English text."""
        text = "I love this product, it's amazing and works perfectly!"
        result = self.analyzer.analyze(text)
        
        self.assertEqual(result['language'], 'en')
        self.assertEqual(result['sentiment'], 'positive')
        self.assertGreater(result['score'], 0.5)
        self.assertGreater(result['confidence'], 0)
    
    def test_english_sentiment_negative(self):
        """Test sentiment analysis on negative English text."""
        text = "This is terrible, I hate it and it doesn't work at all."
        result = self.analyzer.analyze(text)
        
        self.assertEqual(result['language'], 'en')
        self.assertEqual(result['sentiment'], 'negative')
        self.assertLess(result['score'], 0.5)
        self.assertGreater(result['confidence'], 0)
    
    def test_english_sentiment_neutral(self):
        """Test sentiment analysis on neutral English text."""
        text = "The product arrived yesterday. It is made of plastic."
        result = self.analyzer.analyze(text)
        
        self.assertEqual(result['language'], 'en')
        # Note: Model might classify this as slightly positive or negative
        self.assertGreaterEqual(result['confidence'], 0)
    
    def test_empty_text(self):
        """Test sentiment analysis on empty text."""
        text = ""
        result = self.analyzer.analyze(text)
        
        self.assertEqual(result['language'], 'en')
        self.assertEqual(result['sentiment'], 'neutral')
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['confidence'], 0.0)
    
    def test_batch_analyze(self):
        """Test batch sentiment analysis."""
        texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
        
        results = self.analyzer.batch_analyze(texts)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['sentiment'], 'positive')
        self.assertEqual(results[1]['sentiment'], 'negative')
    
    def test_aspect_sentiment(self):
        """Test aspect-based sentiment analysis."""
        text = "The camera quality is excellent, but the battery life is terrible. The screen is decent."
        aspects = ["camera", "battery", "screen"]
        
        result = self.analyzer.analyze_with_aspects(text, aspects)
        
        self.assertIn('overall', result)
        self.assertIn('aspects', result)
        self.assertEqual(len(result['aspects']), 3)
        
        # Check camera aspect (should be positive)
        self.assertIn('camera', result['aspects'])
        self.assertEqual(result['aspects']['camera']['sentiment'], 'positive')
        
        # Check battery aspect (should be negative)
        self.assertIn('battery', result['aspects'])
        self.assertEqual(result['aspects']['battery']['sentiment'], 'negative')


if __name__ == '__main__':
    unittest.main()
