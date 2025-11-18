"""
MLflow utilities for experiment tracking
"""

import mlflow
import mlflow.keras
from typing import Dict, Any
from pathlib import Path
from src.sentiment_analysis.logging.logger import logging as logger


class MLflowTracker:
    """
    Centralized MLflow tracking utilities
    """
    
    def __init__(self, experiment_name: str = "sentiment_analysis", tracking_uri: str = "mlruns"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (local directory or remote server)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: str = None):
        """Start MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters to MLflow")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log single metric"""
        mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log Keras model"""
        mlflow.keras.log_model(model, artifact_path)
        logger.info(f"Model logged to MLflow: {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """Log directory of artifacts"""
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def set_tags(self, tags: Dict[str, Any]):
        """Set run tags"""
        mlflow.set_tags(tags)
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
