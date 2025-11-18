"""
Artifact entities to track outputs from all pipeline components
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class DataIngestionArtifact:
    """Output from data ingestion component"""
    raw_data_path: str
    total_records: int


@dataclass
class DataPreprocessingArtifact:
    """Output from data preprocessing component"""
    processed_data_path: str
    total_records: int
    preprocessing_time_minutes: float


@dataclass
class FeatureEngineeringArtifact:
    """Output from feature engineering component"""
    tokenizer_path: str
    word2vec_path: str
    embedding_matrix_path: str
    vocab_size: int


@dataclass
class ModelTrainerArtifact:
    """Output from model training component"""
    trained_model_path: str          
    checkpoint_path: str             
    training_time_minutes: float
    total_epochs: int
    best_val_accuracy: float
    best_val_loss: float
    final_train_accuracy: float
    final_val_accuracy: float


@dataclass
class ModelEvaluationArtifact:
    """Output from model evaluation component"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    classification_report: Dict
    is_model_accepted: bool
    metrics_file_path: str
    confusion_matrix_path: str


@dataclass
class ModelPusherArtifact:
    """Output from model pusher component"""
    final_model_path: str
    is_model_pushed: bool
    push_timestamp: str


@dataclass
class PredictionArtifact:
    """Output from single prediction"""
    text: str
    cleaned_text: str
    predicted_label: int
    predicted_sentiment: str
    confidence: float
    prediction_probability: float
    processing_time_ms: float


@dataclass
class BatchPredictionArtifact:
    """Output from batch prediction"""
    input_file_path: str
    output_file_path: str
    total_records: int
    successful_predictions: int
    failed_predictions: int
    avg_confidence: float
    processing_time_minutes: float
