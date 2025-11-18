"""
MLflow utilities
"""

import os
import mlflow
import mlflow.keras
from typing import Dict, Any
from src.sentiment_analysis.logging.logger import logging as logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Validate credentials
if not all([tracking_uri, username, password]):
    logger.warning("⚠️  Missing MLFLOW environment variables")
    logger.warning("   MLflow tracking may not work properly")
else:
    # SET ENVIRONMENT VARIABLES IMMEDIATELY (This prevents mlruns creation!)
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = password
    
    # Set tracking URI explicitly
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info(f"✅ MLflow configured: {tracking_uri}")


class MLflowTracker:
    """Simplified MLflow tracker"""
    
    def __init__(self, experiment_name: str = "sentiment_analysis"):
        """Initialize MLflow experiment"""
        self.experiment_name = experiment_name
        
        # Set experiment (will use env vars set above)
        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ MLflow experiment: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, Any] = None):
        """Start MLflow run"""
        run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"✅ Started run: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
        logger.info(f"✅ Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log single metric"""
        mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path: str = "model", 
                  registered_model_name: str = None):
        """Log Keras model"""
        try:
            mlflow.keras.log_model(
                model, 
                artifact_path,
                registered_model_name=registered_model_name
            )
            logger.info(f"✅ Model logged: {artifact_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not log model: {str(e)}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def set_tags(self, tags: Dict[str, Any]):
        """Set tags"""
        mlflow.set_tags(tags)
    
    def end_run(self, status: str = "FINISHED"):
        """End run"""
        mlflow.end_run()
        logger.info(f"✅ Run ended: {status}")
