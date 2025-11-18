"""
MLflow configuration helper.

Reads MLflow-related environment variables and configures the mlflow client.
Recommended env vars (do NOT commit secrets into source control):
- MLFLOW_TRACKING_URI: e.g. https://dagshub.com/<owner>/<repo>.mlflow
- MLFLOW_EXPERIMENT_NAME: optional experiment name (default provided)
- MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD: optional HTTP basic auth
- DAGSHUB_TOKEN: optional personal access token (if your setup requires it)

This module only calls mlflow.set_tracking_uri() and mlflow.set_experiment().
It logs warnings on failure and returns a boolean success flag.
"""

import os
from typing import Optional
import mlflow
from src.sentiment_analysis.logging.logger import logging as logger


def configure_mlflow_from_env(experiment_name: Optional[str] = None) -> bool:
    """Configure MLflow using environment variables.

    Returns True if configuration steps succeeded (or were skipped safely),
    False if a critical configuration error occurred.
    """
    try:
        uri = os.getenv("MLFLOW_TRACKING_URI")
        exp = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment_analysis_lstm_attention")

        if uri:
            try:
                mlflow.set_tracking_uri(uri)
                logger.info(f"MLflow tracking URI set to: {uri}")
            except Exception as e:
                logger.warning(f"Could not set MLflow tracking URI '{uri}': {e}")
                return False
        else:
            logger.info("MLFLOW_TRACKING_URI not set — using MLflow default (local filesystem).")

        try:
            mlflow.set_experiment(exp)
            logger.info(f"MLflow experiment set to: {exp}")
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment '{exp}': {e}")
            return False

        # Log if additional credentials look missing when using DagsHub
        if uri and "dagshub.com" in uri:
            if not (os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN") or os.getenv("MLFLOW_TRACKING_TOKEN")):
                logger.info("Tracking URI points to DagsHub but no auth env vars found. "
                            "If your repo is private, set MLFLOW_TRACKING_USERNAME/MLFLOW_TRACKING_PASSWORD or DAGSHUB_TOKEN in the environment.")

        return True

    except Exception as e:
        logger.warning(f"Unexpected error configuring MLflow: {e}")
        return False
