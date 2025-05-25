"""
Demonstration script for the sentiment analysis component.
This script shows how to use the SentimentAnalyzer for various use cases.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from app.sentiment_analysis.analyzer import SentimentAnalyzer
from app.common.data_utils import load_json_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_sample_text(analyzer: SentimentAnalyzer) -> None:
    """
    Analyze a sample text in different languages.
    
    Args:
        analyzer: SentimentAnalyzer instance
    """
    print("\n=== Sample Text Analysis ===\n")
    
    samples = {
        'en': "I really enjoyed the conference, though the food was mediocre.",
        'es': "El servicio fue excelente pero la comida estaba fría.",
        'fr': "Le film était fantastique, j'ai adoré chaque minute!",
        'de': "Das Produkt hat mich sehr enttäuscht, es funktioniert nicht wie beschrieben.",
        'zh': "这家餐厅的菜很好吃，我一定会再来的！"
    }
    
    for lang, text in samples.items():
        print(f"\nLanguage: {lang}")
        print(f"Text: {text}")
        result = analyzer.analyze(text, language=lang)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 50)


def analyze_sample_dataset(analyzer: SentimentAnalyzer, data_path: str) -> None:
    """
    Analyze a sample dataset of reviews.
    
    Args:
        analyzer: SentimentAnalyzer instance
        data_path: Path to the sample reviews JSON file
    """
    print("\n=== Sample Dataset Analysis ===\n")
    
    try:
        # Load sample reviews
        data = load_json_file(data_path)
        reviews = data.get('reviews', [])
        
        if not reviews:
            print("No reviews found in the dataset.")
            return
        
        print(f"Analyzing {len(reviews)} reviews...")
        
        for review in reviews:
            review_id = review.get('id', 'unknown')
            text = review.get('text', '')
            language = review.get('language', 'en')
            aspects = review.get('aspects', [])
            
            print(f"\nReview #{review_id} ({language}):")
            print(f"Text: {text}")
            
            # Basic sentiment analysis
            result = analyzer.analyze(text, language=language)
            print(f"Overall Sentiment: {result['sentiment']}")
            print(f"Score: {result['score']:.2f}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            # Aspect-based sentiment analysis
            if aspects:
                print("\nAspect Sentiment:")
                aspect_result = analyzer.analyze_with_aspects(text, aspects, language=language)
                
                for aspect, sentiment in aspect_result['aspects'].items():
                    if sentiment['mentions'] > 0:
                        print(f"  - {aspect}: {sentiment['sentiment']} (score: {sentiment['score']:.2f}, mentions: {sentiment['mentions']})")
                    else:
                        print(f"  - {aspect}: not mentioned")
            
            print("-" * 50)
    
    except Exception as e:
        logger.error(f"Error analyzing sample dataset: {str(e)}")
        print(f"Error: {str(e)}")


def batch_analysis_demo(analyzer: SentimentAnalyzer) -> None:
    """
    Demonstrate batch sentiment analysis.
    
    Args:
        analyzer: SentimentAnalyzer instance
    """
    print("\n=== Batch Analysis Demo ===\n")
    
    texts = [
        "The product exceeded all my expectations!",
        "Service was slow and the staff was rude.",
        "It was okay, nothing special but not terrible either.",
        "I've had better experiences elsewhere, wouldn't recommend.",
        "Absolutely fantastic, would buy again in a heartbeat!"
    ]
    
    print(f"Analyzing {len(texts)} texts in batch...")
    results = analyzer.batch_analyze(texts)
    
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 30)


def main():
    """Main function to run the sentiment analysis demo."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Demo')
    parser.add_argument('--data-path', type=str, 
                        default=os.path.join(project_root, 'data', 'sample_reviews.json'),
                        help='Path to sample reviews JSON file')
    parser.add_argument('--language', type=str, default='en',
                        help='Default language for analysis')
    parser.add_argument('--mode', type=str, choices=['all', 'sample', 'dataset', 'batch'],
                        default='all', help='Demo mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize sentiment analyzer
        print("Initializing sentiment analyzer...")
        analyzer = SentimentAnalyzer(default_language=args.language)
        
        # Run demos based on mode
        if args.mode in ['all', 'sample']:
            analyze_sample_text(analyzer)
        
        if args.mode in ['all', 'dataset']:
            analyze_sample_dataset(analyzer, args.data_path)
        
        if args.mode in ['all', 'batch']:
            batch_analysis_demo(analyzer)
            
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in sentiment demo: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
