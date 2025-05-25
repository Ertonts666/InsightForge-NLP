"""
Configuration module for the NLP Insights Engine.
Handles loading configuration from environment variables and config files.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "workers": 1,
        "timeout": 60
    },
    "sentiment_analysis": {
        "default_language": "en",
        "models": {
            "en": "distilbert-base-uncased-finetuned-sst-2-english",
            "multilingual": "nlptown/bert-base-multilingual-uncased-sentiment"
        },
        "batch_size": 32
    },
    "question_answering": {
        "qa_model": "deepset/roberta-base-squad2",
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k_retrieval": 5,
        "threshold": 0.01,
        "vector_db_path": "models/qa_system"
    },
    "storage": {
        "models_dir": "models",
        "data_dir": "data"
    },
    "logging": {
        "level": "INFO",
        "file": None
    }
}


class Config:
    """Configuration manager for the NLP Insights Engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from default, file, and environment variables.
        
        Args:
            config_path: Optional path to JSON configuration file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
        # Ensure paths are absolute
        self._resolve_paths()
        
        logger.info("Configuration loaded")
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Update config recursively
            self._update_dict_recursive(self.config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # API settings
        if os.environ.get('API_HOST'):
            self.config['api']['host'] = os.environ.get('API_HOST')
        if os.environ.get('API_PORT'):
            self.config['api']['port'] = int(os.environ.get('API_PORT'))
        if os.environ.get('API_DEBUG'):
            self.config['api']['debug'] = os.environ.get('API_DEBUG').lower() in ('true', '1', 'yes')
        
        # Sentiment analysis settings
        if os.environ.get('SENTIMENT_DEFAULT_LANGUAGE'):
            self.config['sentiment_analysis']['default_language'] = os.environ.get('SENTIMENT_DEFAULT_LANGUAGE')
        
        # QA settings
        if os.environ.get('QA_MODEL'):
            self.config['question_answering']['qa_model'] = os.environ.get('QA_MODEL')
        if os.environ.get('RETRIEVER_MODEL'):
            self.config['question_answering']['retriever_model'] = os.environ.get('RETRIEVER_MODEL')
        
        # Storage settings
        if os.environ.get('MODELS_DIR'):
            self.config['storage']['models_dir'] = os.environ.get('MODELS_DIR')
        if os.environ.get('DATA_DIR'):
            self.config['storage']['data_dir'] = os.environ.get('DATA_DIR')
        
        # Logging settings
        if os.environ.get('LOG_LEVEL'):
            self.config['logging']['level'] = os.environ.get('LOG_LEVEL')
        if os.environ.get('LOG_FILE'):
            self.config['logging']['file'] = os.environ.get('LOG_FILE')
    
    def _update_dict_recursive(self, target: Dict, source: Dict) -> None:
        """
        Update a dictionary recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict_recursive(target[key], value)
            else:
                target[key] = value
    
    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths."""
        base_dir = Path(__file__).parent.parent.parent.absolute()
        
        # Models directory
        models_dir = self.config['storage']['models_dir']
        if not os.path.isabs(models_dir):
            self.config['storage']['models_dir'] = os.path.join(base_dir, models_dir)
        
        # Data directory
        data_dir = self.config['storage']['data_dir']
        if not os.path.isabs(data_dir):
            self.config['storage']['data_dir'] = os.path.join(base_dir, data_dir)
        
        # Vector DB path
        vector_db_path = self.config['question_answering']['vector_db_path']
        if not os.path.isabs(vector_db_path):
            self.config['question_answering']['vector_db_path'] = os.path.join(
                self.config['storage']['models_dir'], 
                vector_db_path
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., 'api.port')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., 'api.port')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()


# Global configuration instance
config = Config()


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global Config instance
    """
    return config
