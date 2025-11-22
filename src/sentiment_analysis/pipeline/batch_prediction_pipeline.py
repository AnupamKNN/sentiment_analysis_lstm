"""
Batch Prediction Pipeline
"""

import sys
import pandas as pd
import time
import numpy as np
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
            
            # Load data with robust error handling
            # on_bad_lines='skip' ensures one bad row doesn't kill the whole job
            try:
                # Try default UTF-8 first
                df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, trying latin1")
                # Fallback to latin1 for legacy encoding support
                df = pd.read_csv(input_file, encoding='latin1', on_bad_lines='skip')
            except pd.errors.ParserError as e:
                # Catch tokenizing errors (like reading HTML as CSV)
                raise ValueError(f"CSV Parse Error: {str(e)}. Ensure the file is a valid CSV and not a webpage.")
            
            # Normalize headers to handle case sensitivity/whitespace
            df.columns = [c.strip() for c in df.columns]
            
            # Find text column (case insensitive search)
            text_col = next((col for col in df.columns if col.lower() == 'text'), None)
            
            if not text_col:
                raise ValueError(f"Input file must have 'text' column. Found: {list(df.columns)}")
            
            total_records = len(df)
            logger.info(f"Processing {total_records:,} records")
            
            # Lists to store results
            predictions = []
            confidences = []
            probabilities = []  # <--- Store probabilities
            successful = 0
            failed = 0
            
            for idx, text in enumerate(df[text_col], 1):
                try:
                    # Ensure text is string
                    text_content = str(text) if pd.notna(text) else ""
                    
                    result = self.prediction_pipeline.predict(text_content)
                    
                    predictions.append(result.predicted_sentiment)
                    confidences.append(result.confidence)
                    probabilities.append(result.prediction_probability) # <--- Capture Probability
                    successful += 1
                except Exception as e:
                    predictions.append("Error")
                    confidences.append(0.0)
                    probabilities.append(0.0)
                    failed += 1
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{total_records}")
            
            # Add columns to DataFrame
            df['predicted_sentiment'] = predictions
            df['confidence'] = confidences
            df['probability'] = probabilities  # <--- Add to CSV output
            
            # Generate output path if not provided
            if output_file is None:
                output_file = input_file.replace('.csv', '_predictions.csv')
            
            # Ensure directory exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            df.to_csv(output_file, index=False)
            
            processing_time = (time.time() - start_time) / 60
            
            # Calculate Averages
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0
            
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
            
            # Attach avg_probability to artifact for API response
            artifact.avg_probability = avg_probability
            
            logger.info("Batch prediction completed")
            return artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Batch prediction failed: {str(e)}", sys)