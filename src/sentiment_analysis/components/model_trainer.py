"""
Model Trainer Component with MLflow Integration
"""

import sys
import numpy as np
import time
from pathlib import Path
from tensorflow import keras
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import ModelTrainerConfig
from src.sentiment_analysis.entity.artifact_entity import (
    FeatureEngineeringArtifact, ModelTrainerArtifact
)
from src.sentiment_analysis.utils.ml_utils.model.deep_learning_models import LSTMAttentionModel
from src.sentiment_analysis.utils.main_utils.utils import load_numpy_array
from src.sentiment_analysis.utils.ml_utils.mlflow_utils import MLflowTracker  # NEW


class ModelTrainer:
    """Handle model training with MLflow tracking"""
    
    def __init__(self, config: ModelTrainerConfig,
                 feature_artifact: FeatureEngineeringArtifact,
                 use_mlflow: bool = True):  # NEW
        self.config = config
        self.feature_artifact = feature_artifact
        self.use_mlflow = use_mlflow
        
        # Artifact directory
        self.artifact_model_dir = "artifacts/model_trainer/trained_models"
        Path(self.artifact_model_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow tracker
        if self.use_mlflow:
            self.mlflow_tracker = MLflowTracker(
                experiment_name="sentiment_analysis_lstm_attention",
                tracking_uri="mlruns"
            )
        
        logger.info("Model Trainer initialized")
    
    def get_callbacks(self):
        """Create training callbacks"""
        Path(self.config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f"{self.config.checkpoints_dir}/best_model.keras",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add MLflow callback for logging metrics per epoch
        if self.use_mlflow:
            import mlflow.keras
            callbacks_list.append(
                mlflow.keras.MLflowCallback(
                    run_name="lstm_attention_training",
                    log_every_n_steps=1
                )
            )
        
        return callbacks_list
    
    def train_model(self):
        """Train the model with MLflow tracking"""
        try:
            # Start MLflow run
            if self.use_mlflow:
                mlflow_run = self.mlflow_tracker.start_run(
                    run_name=f"lstm_attention_{int(time.time())}"
                )
            
            # Load data
            logger.info("Loading training data...")
            data_path = f"{Path(self.feature_artifact.tokenizer_path).parent}/data_splits.npz"
            data = np.load(data_path)
            
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            
            # Load embedding matrix
            embedding_matrix = load_numpy_array(self.feature_artifact.embedding_matrix_path)
            
            actual_vocab_size = embedding_matrix.shape[0]
            embedding_dim = embedding_matrix.shape[1]
            
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Log parameters to MLflow
            if self.use_mlflow:
                params = {
                    'vocab_size': actual_vocab_size,
                    'embedding_dim': embedding_dim,
                    'max_sequence_length': self.config.max_sequence_length,
                    'lstm_units': self.config.lstm_units,
                    'dropout_rate': self.config.dropout_rate,
                    'dense_units': self.config.dense_units,
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size,
                    'epochs': self.config.epochs,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                }
                self.mlflow_tracker.log_params(params)
                
                # Log tags
                self.mlflow_tracker.set_tags({
                    'model_type': 'LSTM + Attention',
                    'framework': 'TensorFlow/Keras',
                    'task': 'sentiment_classification',
                    'dataset': 'Sentiment140'
                })
            
            # Build model
            logger.info("Building model...")
            model = LSTMAttentionModel.build(
                embedding_matrix=embedding_matrix,
                vocab_size=actual_vocab_size,
                max_length=self.config.max_sequence_length,
                embedding_dim=embedding_dim,
                lstm_units=self.config.lstm_units,
                dropout_rate=self.config.dropout_rate,
                dense_units=self.config.dense_units,
                learning_rate=self.config.learning_rate
            )
            
            logger.info(f"Model parameters: {model.count_params():,}")
            
            # Log model summary to MLflow
            if self.use_mlflow:
                self.mlflow_tracker.log_metric('total_parameters', model.count_params())
            
            # Train
            logger.info("Starting training...")
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=self.get_callbacks(),
                verbose=1
            )
            
            training_time = (time.time() - start_time) / 60
            
            # Log final metrics to MLflow
            if self.use_mlflow:
                final_metrics = {
                    'final_train_loss': history.history['loss'][-1],
                    'final_train_accuracy': history.history['accuracy'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'best_val_accuracy': max(history.history['val_accuracy']),
                    'training_time_minutes': training_time,
                    'total_epochs': len(history.history['loss'])
                }
                self.mlflow_tracker.log_metrics(final_metrics)
            
            # Save models
            artifact_model_path = f"{self.artifact_model_dir}/lstm_attention_model.keras"
            model.save(artifact_model_path)
            logger.info(f"✅ Model saved to artifacts: {artifact_model_path}")
            
            Path(self.config.trained_model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(self.config.trained_model_path)
            logger.info(f"✅ Model saved to production: {self.config.trained_model_path}")
            
            # Log model to MLflow
            if self.use_mlflow:
                self.mlflow_tracker.log_model(model, artifact_path="model")
                
                # Log model files as artifacts
                self.mlflow_tracker.log_artifact(artifact_model_path, "saved_models")
            
            logger.info(f"Training completed in {training_time:.2f} minutes")
            
            # End MLflow run
            if self.use_mlflow:
                self.mlflow_tracker.end_run()
            
            return {
                'model': model,
                'history': history,
                'training_time': training_time,
                'artifact_model_path': artifact_model_path,
                'production_model_path': self.config.trained_model_path
            }
        
        except Exception as e:
            # End MLflow run on error
            if self.use_mlflow:
                self.mlflow_tracker.end_run()
            raise SentimentAnalysisException(f"Training failed: {str(e)}", sys)
    
    # ... rest of the methods remain same ...
