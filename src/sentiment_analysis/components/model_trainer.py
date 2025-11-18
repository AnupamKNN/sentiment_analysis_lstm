import os
import sys
import time
import numpy as np
from pathlib import Path
from tensorflow import keras
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import ModelTrainerConfig
from src.sentiment_analysis.entity.artifact_entity import FeatureEngineeringArtifact, ModelTrainerArtifact
from src.sentiment_analysis.utils.main_utils.utils import load_numpy_array
from src.sentiment_analysis.utils.ml_utils.metric.metric import get_classification_score
from src.sentiment_analysis.utils.ml_utils.model.deep_learning_models import LSTMAttentionModel
from src.sentiment_analysis.constants import ARTIFACTS_DIR
import mlflow
from dotenv import load_dotenv

load_dotenv()  # Load env variables early

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig,
                 feature_artifact: FeatureEngineeringArtifact,
                 use_mlflow: bool = True):
        self.config = config
        self.feature_artifact = feature_artifact
        self.use_mlflow = use_mlflow
        self.artifact_model_dir = "Artifacts/model_trainer/trained_models"
        Path(self.artifact_model_dir).mkdir(parents=True, exist_ok=True)

        if self.use_mlflow:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            username = os.getenv("MLFLOW_TRACKING_USERNAME")
            password = os.getenv("MLFLOW_TRACKING_PASSWORD")

            os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = password

            mlflow.set_tracking_uri(tracking_uri)

    def track_mlflow(self, metrics: dict):
        with mlflow.start_run(run_name="Sentiment_LSTM_Attention_Model"):
            # Filter out None metrics e.g. auc_roc if missing
            filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
            for key, val in filtered_metrics.items():
                mlflow.log_metric(key, val)
            logger.info(f"Metrics tracked in MLflow: {filtered_metrics}")

    def train_model(self):
        try:
            logger.info("Loading training and test data splits...")
            data_path = os.path.join(ARTIFACTS_DIR, "feature_engineering", "data_splits.npz")
            data = np.load(data_path)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            X_test, y_test = data['X_test'], data['y_test']

            logger.info(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}, Test samples: {len(X_test):,}")

            embedding_matrix = load_numpy_array(self.feature_artifact.embedding_matrix_path)
            vocab_size = embedding_matrix.shape[0]
            embedding_dim = embedding_matrix.shape[1]

            logger.info("Building model...")
            model = LSTMAttentionModel.build(
                embedding_matrix=embedding_matrix,
                vocab_size=vocab_size,
                max_length=self.config.max_sequence_length,
                embedding_dim=embedding_dim,
                lstm_units=self.config.lstm_units,
                dropout_rate=self.config.dropout_rate,
                dense_units=self.config.dense_units,
                learning_rate=self.config.learning_rate
            )

            logger.info("Training model...")
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
            logger.info(f"Training completed in {training_time:.2f} minutes")

            # Evaluate on test data
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            metrics = get_classification_score(y_test, y_pred, y_pred_proba)
            logger.info(f"Test metrics: {metrics}")

            if self.use_mlflow:
                self.track_mlflow(metrics)

            # Saving models
            artifact_model_path = f"{self.artifact_model_dir}/lstm_attention_model.keras"
            model.save(artifact_model_path)
            logger.info(f"Model saved to artifact path: {artifact_model_path}")

            final_model_path = self.config.trained_model_path
            Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(final_model_path)
            logger.info(f"Model saved to production path: {final_model_path}")

            return {
                'model': model,
                'history': history,
                'training_time': training_time,
                'artifact_model_path': artifact_model_path,
                'production_model_path': final_model_path,
                'test_metrics': metrics
            }
        except Exception as e:
            raise SentimentAnalysisException(f"Training failed: {str(e)}", sys)

    def get_callbacks(self):
        Path(self.config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        return [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(f"{self.config.checkpoints_dir}/best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        ]

    def initiate_model_training(self) -> ModelTrainerArtifact:
        res = self.train_model()
        history = res['history']
        artifact = ModelTrainerArtifact(
            trained_model_path=res['production_model_path'],
            checkpoint_path=f"{self.config.checkpoints_dir}/best_model.keras",
            training_time_minutes=res['training_time'],
            total_epochs=len(history.history['loss']),
            best_val_accuracy=max(history.history['val_accuracy']),
            best_val_loss=min(history.history['val_loss']),
            final_train_accuracy=history.history['accuracy'][-1],
            final_val_accuracy=history.history['val_accuracy'][-1]
        )
        logger.info(f"Model training artifact: {artifact}")
        return artifact
