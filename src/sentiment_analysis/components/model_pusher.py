"""
Model Pusher Component
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.artifact_entity import ModelPusherArtifact


class ModelPusher:
    """Push model to production directory"""
    
    def __init__(self, model_path: str, final_path: str):
        self.model_path = model_path
        self.final_path = final_path
        logger.info("Model Pusher initialized")
    
    def push_model(self) -> ModelPusherArtifact:
        """Push model to production"""
        try:
            logger.info("Pushing model to production...")
            
            Path(self.final_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.model_path, self.final_path)
            
            logger.info(f"Model pushed: {self.final_path}")
            
            return ModelPusherArtifact(
                final_model_path=self.final_path,
                is_model_pushed=True,
                push_timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            raise SentimentAnalysisException(f"Model push failed: {str(e)}", sys)
