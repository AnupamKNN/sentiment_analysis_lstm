"""
Configuration entities
"""

from dataclasses import dataclass
import os
from src.sentiment_analysis.constants import *
from src.sentiment_analysis.constants.model_constants import *


@dataclass
class DataIngestionConfig:
    dataset_slug: str = DATASET_SLUG
    raw_data_dir: str = RAW_DATA_DIR
    artifacts_dir: str = ARTIFACTS_DIR


@dataclass
class DataPreprocessingConfig:
    processed_data_dir: str = PROCESSED_DATA_DIR
    chunk_size: int = 100000
    text_column: str = TEXT_COLUMN
    sentiment_column: str = SENTIMENT_COLUMN


@dataclass
class FeatureEngineeringConfig:
    max_vocab_size: int = MAX_VOCAB_SIZE
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    tokenizer_path: str = os.path.join(FINAL_VECTORIZERS_DIR, TOKENIZER_FILENAME)
    word2vec_path: str = os.path.join(FINAL_VECTORIZERS_DIR, WORD2VEC_FILENAME)
    embedding_matrix_path: str = os.path.join(EMBEDDING_MATRIX_DIR, EMBEDDING_MATRIX_FILENAME)


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join(FINAL_TRAINED_MODELS_DIR, MODEL_FILENAME)
    checkpoints_dir: str = CHECKPOINTS_DIR
    lstm_units: int = LSTM_UNITS
    dropout_rate: float = DROPOUT_RATE
    dense_units: int = DENSE_UNITS
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    embedding_dim: int = EMBEDDING_DIM  


@dataclass
class PredictionPipelineConfig:
    model_path: str = os.path.join(FINAL_TRAINED_MODELS_DIR, MODEL_FILENAME)
    tokenizer_path: str = os.path.join(FINAL_VECTORIZERS_DIR, TOKENIZER_FILENAME)
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    prediction_threshold: float = PREDICTION_THRESHOLD