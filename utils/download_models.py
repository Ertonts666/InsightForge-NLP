"""
Utility script to download and prepare models for the NLP Insights Engine.
This script downloads the required models for sentiment analysis and question answering.
"""

import os
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
SENTIMENT_MODELS = {
    'en': 'distilbert-base-uncased-finetuned-sst-2-english',
    'multilingual': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'zh': 'uer/roberta-base-finetuned-jd-binary-chinese',
    'fr': 'tblard/tf-allocine',
    'de': 'oliverguhr/german-sentiment-bert',
    'es': 'dccuchile/bert-base-spanish-wwm-uncased-sentiment',
}

QA_MODELS = {
    'default': 'deepset/roberta-base-squad2',
}

EMBEDDING_MODELS = {
    'default': 'sentence-transformers/all-MiniLM-L6-v2',
}


def download_sentiment_models(models_dir: str, languages: list = None) -> None:
    """
    Download sentiment analysis models.
    
    Args:
        models_dir: Directory to save models
        languages: List of language codes to download (if None, download all)
    """
    os.makedirs(os.path.join(models_dir, 'sentiment'), exist_ok=True)
    
    # Determine which languages to download
    if languages:
        models_to_download = {lang: model for lang, model in SENTIMENT_MODELS.items() if lang in languages}
    else:
        models_to_download = SENTIMENT_MODELS
    
    # Download models
    for lang, model_name in models_to_download.items():
        logger.info(f"Downloading sentiment model for {lang}: {model_name}")
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Download model
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Save model and tokenizer
            model_dir = os.path.join(models_dir, 'sentiment', lang)
            os.makedirs(model_dir, exist_ok=True)
            
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            
            logger.info(f"Successfully downloaded and saved sentiment model for {lang}")
        except Exception as e:
            logger.error(f"Error downloading sentiment model for {lang}: {str(e)}")


def download_qa_models(models_dir: str) -> None:
    """
    Download question answering models.
    
    Args:
        models_dir: Directory to save models
    """
    os.makedirs(os.path.join(models_dir, 'qa'), exist_ok=True)
    
    # Download models
    for model_type, model_name in QA_MODELS.items():
        logger.info(f"Downloading QA model {model_type}: {model_name}")
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Download model
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Save model and tokenizer
            model_dir = os.path.join(models_dir, 'qa', model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            
            logger.info(f"Successfully downloaded and saved QA model {model_type}")
        except Exception as e:
            logger.error(f"Error downloading QA model {model_type}: {str(e)}")


def download_embedding_models(models_dir: str) -> None:
    """
    Download embedding models for retrieval.
    
    Args:
        models_dir: Directory to save models
    """
    os.makedirs(os.path.join(models_dir, 'embeddings'), exist_ok=True)
    
    # Download models
    for model_type, model_name in EMBEDDING_MODELS.items():
        logger.info(f"Downloading embedding model {model_type}: {model_name}")
        try:
            # Download model
            model = SentenceTransformer(model_name)
            
            # Save model
            model_dir = os.path.join(models_dir, 'embeddings', model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            model.save(model_dir)
            
            logger.info(f"Successfully downloaded and saved embedding model {model_type}")
        except Exception as e:
            logger.error(f"Error downloading embedding model {model_type}: {str(e)}")


def main():
    """Main function to download all required models."""
    parser = argparse.ArgumentParser(description='Download models for NLP Insights Engine')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--sentiment-only', action='store_true',
                        help='Download only sentiment models')
    parser.add_argument('--qa-only', action='store_true',
                        help='Download only QA models')
    parser.add_argument('--embeddings-only', action='store_true',
                        help='Download only embedding models')
    parser.add_argument('--languages', type=str, nargs='+',
                        help='Languages to download sentiment models for')
    
    args = parser.parse_args()
    
    # Resolve models directory path
    models_dir = args.models_dir
    if not os.path.isabs(models_dir):
        # Get the project root directory (parent of the directory containing this script)
        project_root = Path(__file__).parent.parent.absolute()
        models_dir = os.path.join(project_root, models_dir)
    
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Downloading models to {models_dir}")
    
    # Download models based on arguments
    if args.sentiment_only:
        download_sentiment_models(models_dir, args.languages)
    elif args.qa_only:
        download_qa_models(models_dir)
    elif args.embeddings_only:
        download_embedding_models(models_dir)
    else:
        # Download all models
        download_sentiment_models(models_dir, args.languages)
        download_qa_models(models_dir)
        download_embedding_models(models_dir)
    
    logger.info("Model download complete")


if __name__ == "__main__":
    main()
