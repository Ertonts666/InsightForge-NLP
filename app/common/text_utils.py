"""
Text preprocessing utilities for the NLP Insights Engine.
Provides functions for cleaning, tokenizing, and processing text data.
"""

import re
import string
import logging
from typing import List, Dict, Set, Optional, Tuple
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common stopwords for multiple languages
STOPWORDS = {
    'en': {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'who', 'which'},
    'es': {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'si', 'porque', 'como', 'qué', 'cuándo'},
    'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'si', 'parce', 'que', 'quand', 'où', 'comment'},
    'de': {'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'wenn', 'weil', 'als', 'was', 'wann', 'wo', 'wie'},
    'zh': {'的', '了', '和', '是', '在', '我', '有', '不', '这', '为', '也', '你', '都', '他', '么'}
}


def normalize_text(text: str, lowercase: bool = True, remove_accents: bool = False) -> str:
    """
    Normalize text by lowercasing and optionally removing accents.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_accents: Whether to remove accents
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove accents if requested
    if remove_accents:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(tokens: List[str], language: str = 'en', custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens: List of tokens
        language: Language code
        custom_stopwords: Optional set of custom stopwords to remove
        
    Returns:
        List of tokens without stopwords
    """
    # Get stopwords for the language
    stopwords = STOPWORDS.get(language, set())
    
    # Add custom stopwords if provided
    if custom_stopwords:
        stopwords = stopwords.union(custom_stopwords)
    
    # Remove stopwords
    return [token for token in tokens if token.lower() not in stopwords]


def tokenize_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_punct: Whether to remove punctuation
        
    Returns:
        List of tokens
    """
    # Normalize text
    if lowercase:
        text = text.lower()
    
    # Remove punctuation if requested
    if remove_punct:
        text = remove_punctuation(text)
    
    # Tokenize
    tokens = text.split()
    
    return tokens


def clean_text(
    text: str, 
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_accents: bool = False,
    remove_digits: bool = False,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_extra_spaces: bool = True
) -> str:
    """
    Clean text by applying multiple preprocessing steps.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_punct: Whether to remove punctuation
        remove_accents: Whether to remove accents
        remove_digits: Whether to remove digits
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        remove_extra_spaces: Whether to remove extra whitespace
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    if remove_emails:
        text = re.sub(r'\S+@\S+', '', text)
    
    # Remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    
    # Normalize text
    text = normalize_text(text, lowercase, remove_accents)
    
    # Remove punctuation
    if remove_punct:
        text = remove_punctuation(text)
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = remove_extra_whitespace(text)
    
    return text


def split_into_sentences(text: str, language: str = 'en') -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        language: Language code
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting for English and similar languages
    if language in ['en', 'es', 'fr', 'de']:
        # Split on sentence-ending punctuation followed by space and uppercase letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Handle cases where there's no space after punctuation
        result = []
        for sentence in sentences:
            # Further split on sentence-ending punctuation if needed
            subsents = re.split(r'(?<=[.!?])(?=[A-Z])', sentence)
            result.extend(subsents)
        
        return [s.strip() for s in result if s.strip()]
    
    # For Chinese, split on common sentence-ending punctuation
    elif language == 'zh':
        return [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
    
    # Default: simple split on period, exclamation, question mark
    else:
        return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def extract_keywords(text: str, language: str = 'en', top_n: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency-based approach.
    
    Args:
        text: Input text
        language: Language code
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Clean and tokenize text
    clean = clean_text(text, lowercase=True, remove_punct=True)
    tokens = tokenize_text(clean)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens, language)
    
    # Count token frequencies
    token_counts = {}
    for token in tokens:
        if len(token) > 2:  # Only consider tokens with more than 2 characters
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Sort by frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [token for token, count in sorted_tokens[:top_n]]


def detect_language(text: str) -> str:
    """
    Detect the language of a text using a simple heuristic approach.
    
    Args:
        text: Input text
        
    Returns:
        Detected language code
    """
    # This is a very simple implementation
    # In a production system, use a proper language detection library like langdetect
    
    # Count stopwords from each language
    counts = {}
    
    # Clean and tokenize text
    clean = clean_text(text, lowercase=True, remove_punct=True)
    tokens = tokenize_text(clean)
    
    # Count stopwords for each language
    for lang, stopwords in STOPWORDS.items():
        counts[lang] = sum(1 for token in tokens if token in stopwords)
    
    # Return language with most stopwords
    if not counts or max(counts.values()) == 0:
        return 'en'  # Default to English if no stopwords found
    
    return max(counts.items(), key=lambda x: x[1])[0]
