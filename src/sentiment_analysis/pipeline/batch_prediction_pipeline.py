"""
Batch Prediction Pipeline
"""

import sys
import pandas as pd
import time
from pathlib import Path
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.pipeline.prediction_pipeline import PredictionPipeline
from src.sentiment_analysis.entity.artifact_entity import BatchPredictionArtifact


class BatchPredictionPipeline:
    """Batch prediction pipeline for multiple texts"""
    
    def __init__(self):
        self.prediction_pipeline = PredictionPipeline()
        logger.info("Batch Prediction Pipeline initialized")
    
    def predict_batch(self, input_file: str, output_file: str = None) -> BatchPredictionArtifact:
        """Predict for batch of texts from CSV"""
        try:
            logger.info(f"Loading batch data from: {input_file}")
            start_time = time.time()
            
            # Load data
            df = pd.read_csv(input_file)
            
            if 'text' not in df.columns:
                raise ValueError("Input file must have 'text' column")
            
            total_records = len(df)
            logger.info(f"Processing {total_records:,} records")
            
            # Predict for each text
            predictions = []
            confidences = []
            successful = 0
            failed = 0
            
            for idx, text in enumerate(df['text'], 1):
                try:
                    result = self.prediction_pipeline.predict(text)
                    predictions.append(result.predicted_sentiment)
                    confidences.append(result.confidence)
                    successful += 1
                except:
                    predictions.append("Error")
                    confidences.append(0.0)
                    failed += 1
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{total_records}")
            
            # Add predictions to dataframe
            df['predicted_sentiment'] = predictions
            df['confidence'] = confidences
            
            # Save results
            if output_file is None:
                output_file = input_file.replace('.csv', '_predictions.csv')
            
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            
            processing_time = (time.time() - start_time) / 60
            avg_confidence = sum([c for c in confidences if c > 0]) / successful if successful > 0 else 0
            
            logger.info(f"Batch predictions saved: {output_file}")
            
            artifact = BatchPredictionArtifact(
                input_file_path=input_file,
                output_file_path=output_file,
                total_records=total_records,
                successful_predictions=successful,
                failed_predictions=failed,
                avg_confidence=avg_confidence,
                processing_time_minutes=processing_time
            )
            
            logger.info("Batch prediction completed")
            return artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Batch prediction failed: {str(e)}", sys)
