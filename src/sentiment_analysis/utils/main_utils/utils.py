"""
Main utility functions
"""

import os
import sys
import yaml
import pickle
import numpy as np
from typing import Any, Dict
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException


def read_yaml(file_path: str) -> Dict:
    """Read YAML file"""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise SentimentAnalysisException(f"Error reading YAML: {str(e)}", sys)


def save_object(file_path: str, obj: Any):
    """Save Python object as pickle"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logger.info(f"Object saved: {file_path}")
    except Exception as e:
        raise SentimentAnalysisException(f"Error saving object: {str(e)}", sys)


def load_object(file_path: str) -> Any:
    """Load pickled object"""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise SentimentAnalysisException(f"Error loading object: {str(e)}", sys)


def save_numpy_array(file_path: str, array: np.ndarray):
    """Save numpy array"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
        logger.info(f"Array saved: {file_path}")
    except Exception as e:
        raise SentimentAnalysisException(f"Error saving array: {str(e)}", sys)


def load_numpy_array(file_path: str) -> np.ndarray:
    """Load numpy array"""
    try:
        return np.load(file_path)
    except Exception as e:
        raise SentimentAnalysisException(f"Error loading array: {str(e)}", sys)


def create_directories(directories: list):
    """Create multiple directories"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
