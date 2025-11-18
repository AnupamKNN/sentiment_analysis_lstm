"""
Main entry point for training pipeline
"""

from src.sentiment_analysis.pipeline.training_pipeline import TrainingPipeline
from src.sentiment_analysis.logging.logger import logging as logger

if __name__ == "__main__":
    try:
        # Initialize and run training pipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        
        logger.info("Training completed successfully!")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
