"""
Project-wide constants
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# CRITICAL: Set MLflow env vars immediately
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set as environment variables RIGHT NOW
if MLFLOW_TRACKING_URI:
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
if MLFLOW_TRACKING_USERNAME:
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
if MLFLOW_TRACKING_PASSWORD:
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "Artifacts/data_ingestion")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "Artifacts/model_trainer")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Artifacts directories
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "Artifacts")
TRAINED_MODELS_DIR = os.path.join(ARTIFACTS_DIR, "model_trainer/trained_models")

# Final models directory (for production)
FINAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "final_models")
FINAL_TRAINED_MODELS_DIR = os.path.join(FINAL_MODELS_DIR, "trained_models")
FINAL_VECTORIZERS_DIR = os.path.join(FINAL_MODELS_DIR, "vectorizers")
FINAL_PREPROCESSING_DIR = os.path.join(FINAL_MODELS_DIR, "preprocessing")
EMBEDDING_MATRIX_DIR = os.path.join(ARTIFACTS_DIR, "preprocessing")

# Logs directory
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Results directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "training_results")
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")

# Dataset configuration
DATASET_SLUG = "kazanova/sentiment140"
RAW_DATA_FILENAME = "training.1600000.processed.noemoticon.csv"

# Column names
SENTIMENT_COLUMN = "sentiment"
TEXT_COLUMN = "text"
TEXT_CLEANED_COLUMN = "text_cleaned"
TOKENS_COLUMN = "tokens"
TOKENS_STR_COLUMN = "tokens_str"
TOKEN_COUNT_COLUMN = "token_count"

# Data schema
DATASET_COLUMNS = ['sentiment', 'id', 'date', 'query', 'user', 'text']