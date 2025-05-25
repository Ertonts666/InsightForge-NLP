"""
Data processing utilities for the NLP Insights Engine.
Provides functions for loading, processing, and managing datasets.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        encoding: File encoding
        
    Returns:
        File contents as string
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading text file {file_path}: {str(e)}")
        raise


def load_json_file(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        encoding: File encoding
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise


def save_json_file(data: Dict[str, Any], file_path: str, encoding: str = 'utf-8', indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        encoding: File encoding
        indent: JSON indentation level
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(f"Saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise


def load_csv_file(
    file_path: str, 
    encoding: str = 'utf-8',
    delimiter: str = ',',
    quotechar: str = '"',
    as_dict: bool = True
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        encoding: File encoding
        delimiter: CSV delimiter
        quotechar: CSV quote character
        as_dict: Whether to return as list of dictionaries (True) or pandas DataFrame (False)
        
    Returns:
        CSV data as list of dictionaries or pandas DataFrame
    """
    try:
        if as_dict:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
                return list(reader)
        else:
            return pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, quotechar=quotechar)
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        raise


def save_csv_file(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    file_path: str,
    encoding: str = 'utf-8',
    delimiter: str = ',',
    quotechar: str = '"'
) -> None:
    """
    Save data to a CSV file.
    
    Args:
        data: Data to save (list of dictionaries or pandas DataFrame)
        file_path: Path to save the CSV file
        encoding: File encoding
        delimiter: CSV delimiter
        quotechar: CSV quote character
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, encoding=encoding, index=False, sep=delimiter, quotechar=quotechar)
        else:
            if not data:
                logger.warning(f"Empty data provided for CSV file {file_path}")
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write('')
                return
            
            fieldnames = data[0].keys()
            with open(file_path, 'w', encoding=encoding, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar)
                writer.writeheader()
                writer.writerows(data)
        
        logger.info(f"Saved CSV data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {str(e)}")
        raise


def batch_generator(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Generate batches from a list of data.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        
    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def load_dataset(
    file_path: str,
    format: str = None,
    encoding: str = 'utf-8',
    **kwargs
) -> Union[str, Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """
    Load a dataset from a file.
    
    Args:
        file_path: Path to the dataset file
        format: File format ('json', 'csv', 'txt', or None to infer from extension)
        encoding: File encoding
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        Loaded dataset
    """
    # Infer format from file extension if not provided
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lstrip('.').lower()
    
    # Load based on format
    if format == 'json':
        return load_json_file(file_path, encoding=encoding)
    elif format == 'csv':
        return load_csv_file(file_path, encoding=encoding, **kwargs)
    elif format == 'txt':
        return load_text_file(file_path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported file format: {format}")


def save_dataset(
    data: Any,
    file_path: str,
    format: str = None,
    encoding: str = 'utf-8',
    **kwargs
) -> None:
    """
    Save a dataset to a file.
    
    Args:
        data: Data to save
        file_path: Path to save the dataset file
        format: File format ('json', 'csv', 'txt', or None to infer from extension)
        encoding: File encoding
        **kwargs: Additional arguments for specific savers
    """
    # Infer format from file extension if not provided
    if format is None:
        _, ext = os.path.splitext(file_path)
        format = ext.lstrip('.').lower()
    
    # Save based on format
    if format == 'json':
        save_json_file(data, file_path, encoding=encoding, **kwargs)
    elif format == 'csv':
        save_csv_file(data, file_path, encoding=encoding, **kwargs)
    elif format == 'txt':
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(data)
            logger.info(f"Saved text data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving text file {file_path}: {str(e)}")
            raise
    else:
        raise ValueError(f"Unsupported file format: {format}")


def split_dataset(
    data: Union[List[Any], pd.DataFrame],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        data: Dataset to split
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        shuffle: Whether to shuffle the data before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    import numpy as np
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Convert pandas DataFrame to list if needed
    if isinstance(data, pd.DataFrame):
        data_list = data.to_dict('records')
    else:
        data_list = data
    
    # Shuffle data if requested
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(data_list)
    
    # Calculate split indices
    n = len(data_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split data
    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]
    
    # Convert back to DataFrame if input was DataFrame
    if isinstance(data, pd.DataFrame):
        train_data = pd.DataFrame(train_data)
        val_data = pd.DataFrame(val_data)
        test_data = pd.DataFrame(test_data)
    
    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    return train_data, val_data, test_data
