"""
Data Preprocessing Component
"""

import sys
import pandas as pd
import time
from pathlib import Path
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import DataPreprocessingConfig
from src.sentiment_analysis.entity.artifact_entity import (
    DataIngestionArtifact, DataPreprocessingArtifact
)
from src.sentiment_analysis.utils.ml_utils.preprocessing import TextPreprocessor
from src.sentiment_analysis.constants import SENTIMENT_COLUMN, TEXT_COLUMN


class DataPreprocessing:
    """Handle data preprocessing"""
    
    def __init__(self, config: DataPreprocessingConfig, 
                 data_ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact
        logger.info("Data Preprocessing initialized")
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess raw data"""
        try:
            logger.info("Loading raw data...")
            
            # FIX: Use latin-1 encoding for Sentiment140 dataset
            df = pd.read_csv(
                self.data_ingestion_artifact.raw_data_path,
                encoding='latin-1',  # ← CRITICAL FIX
                header=None,
                names=['sentiment', 'id', 'date', 'query', 'user', 'text']
            )
            
            logger.info(f"Processing {len(df):,} records")
            
            # Clean text
            logger.info("Cleaning text...")
            df = TextPreprocessor.clean_dataframe(df, self.config.text_column)
            
            # Tokenize
            logger.info("Tokenizing...")
            df = TextPreprocessor.tokenize_dataframe(
                df, 
                text_column='text_cleaned',
                chunk_size=self.config.chunk_size
            )
            
            # Keep only required columns
            required_cols = [
                'sentiment', 'text', 'text_cleaned',
                'tokens_str', 'token_count'
            ]
            df = df[required_cols]
            
            logger.info("Preprocessing completed")
            return df
        
        except Exception as e:
            raise SentimentAnalysisException(f"Preprocessing failed: {str(e)}", sys)
    
    def initiate_data_preprocessing(self) -> DataPreprocessingArtifact:
        """Execute preprocessing"""
        try:
            logger.info("Starting data preprocessing")
            start_time = time.time()
            
            # Preprocess
            df = self.preprocess_data()
            
            # Save
            Path(self.config.processed_data_dir).mkdir(parents=True, exist_ok=True)
            processed_path = f"{self.config.processed_data_dir}/sentiment140_processed.csv"
            
            logger.info(f"Saving processed data to: {processed_path}")
            df.to_csv(processed_path, index=False, encoding='utf-8')
            
            processing_time = (time.time() - start_time) / 60
            
            logger.info(f"Processed data saved: {processed_path}")
            logger.info(f"Processing time: {processing_time:.2f} minutes")
            
            artifact = DataPreprocessingArtifact(
                processed_data_path=processed_path,
                total_records=len(df),
                preprocessing_time_minutes=processing_time
            )
            
            logger.info("Data preprocessing completed")
            return artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Preprocessing failed: {str(e)}", sys)