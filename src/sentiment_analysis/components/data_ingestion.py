"""
Data Ingestion Component
"""

import os
import sys
import shutil
import kagglehub
import pandas as pd
from pathlib import Path
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import DataIngestionConfig
from src.sentiment_analysis.entity.artifact_entity import DataIngestionArtifact
from src.sentiment_analysis.constants import DATASET_COLUMNS


class DataIngestion:
    """Handle data ingestion from Kaggle with smart caching"""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.expected_filename = "training.1600000.processed.noemoticon.csv"
        logger.info("Data Ingestion initialized")
    
    def check_existing_data(self) -> str:
        """
        Check if dataset already exists locally
        
        Returns:
            Path to existing file or None if not found
        """
        try:
            # Check for expected filename
            expected_path = os.path.join(self.config.raw_data_dir, self.expected_filename)
            
            if os.path.exists(expected_path):
                file_size = os.path.getsize(expected_path) / (1024 * 1024)  # MB
                logger.info(f"✅ Dataset found: {expected_path}")
                logger.info(f"   File size: {file_size:.2f} MB")
                
                # Verify it's not corrupted by checking size (Sentiment140 is ~80MB)
                if file_size > 70:
                    logger.info("   ✅ File size valid - skipping download")
                    return expected_path
                else:
                    logger.warning(f"   ⚠️ File size too small ({file_size:.2f} MB) - may be corrupted")
                    return None
            
            # Check for any CSV files in raw data directory
            if os.path.exists(self.config.raw_data_dir):
                csv_files = list(Path(self.config.raw_data_dir).glob("*.csv"))
                
                if csv_files:
                    file_path = str(csv_files[0])
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"✅ Found CSV file: {file_path}")
                    logger.info(f"   File size: {file_size:.2f} MB")
                    
                    if file_size > 70:
                        logger.info("   ✅ Using existing CSV file")
                        return file_path
                    else:
                        logger.warning(f"   ⚠️ File size too small - may be corrupted")
            
            logger.info("❌ No valid dataset found locally")
            return None
        
        except Exception as e:
            logger.error(f"Error checking existing data: {str(e)}")
            return None
    
    def download_dataset(self) -> str:
        """Download Sentiment140 from Kaggle"""
        try:
            logger.info("="*60)
            logger.info("📥 DOWNLOADING DATASET FROM KAGGLE")
            logger.info("="*60)
            logger.info(f"Dataset: {self.config.dataset_slug}")
            logger.info("This may take a few minutes...")
            
            # Download using kagglehub
            download_path = kagglehub.dataset_download(self.config.dataset_slug)
            logger.info(f"✅ Downloaded to: {download_path}")
            
            # Create destination directory
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            
            # Find CSV file in download path
            csv_files = list(Path(download_path).glob("*.csv"))
            
            if not csv_files:
                raise Exception(f"No CSV files found in {download_path}")
            
            # Copy first CSV to raw data directory with expected name
            source_file = csv_files[0]
            dest_file = os.path.join(self.config.raw_data_dir, self.expected_filename)
            
            logger.info(f"📋 Copying dataset...")
            logger.info(f"   From: {source_file}")
            logger.info(f"   To:   {dest_file}")
            
            shutil.copy2(source_file, dest_file)
            
            # Get file size
            file_size = os.path.getsize(dest_file) / (1024 * 1024)
            logger.info(f"✅ Dataset ready!")
            logger.info(f"   Location: {dest_file}")
            logger.info(f"   Size: {file_size:.2f} MB")
            logger.info("="*60)
            
            return dest_file
        
        except Exception as e:
            raise SentimentAnalysisException(f"Download failed: {str(e)}", sys)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate raw CSV"""
        try:
            logger.info("📖 Loading raw data...")
            
            # Load CSV
            df = pd.read_csv(
                file_path,
                encoding='latin-1',
                header=None,
                names=DATASET_COLUMNS,
                dtype={'sentiment': 'int8', 'id': 'int64'}
            )
            
            logger.info(f"✅ Loaded {len(df):,} records")
            
            # Quick validation
            logger.info("🔍 Validating dataset...")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info(f"   Shape: {df.shape}")
            logger.info(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Check sentiment distribution
            sentiment_dist = df['sentiment'].value_counts()
            logger.info(f"   Sentiment distribution:")
            for label, count in sentiment_dist.items():
                logger.info(f"      {label}: {count:,} ({count/len(df)*100:.1f}%)")
            
            return df
        
        except Exception as e:
            raise SentimentAnalysisException(f"Loading failed: {str(e)}", sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute data ingestion with smart caching
        
        Flow:
        1. Check if data exists locally
        2. If exists and valid, use it (skip download)
        3. If not exists, download from Kaggle
        4. Load and validate data
        5. Return artifact
        """
        try:
            logger.info("="*70)
            logger.info("🚀 STARTING DATA INGESTION")
            logger.info("="*70)
            
            # Step 1: Check for existing data
            logger.info("📂 Step 1: Checking for existing dataset...")
            existing_path = self.check_existing_data()
            
            if existing_path:
                # Data already exists - skip download
                logger.info("✅ Using existing dataset - skipping download")
                raw_data_path = existing_path
            else:
                # Download from Kaggle
                logger.info("📥 Step 2: Downloading dataset from Kaggle...")
                raw_data_path = self.download_dataset()
            
            # Step 3: Load and validate
            logger.info("📖 Step 3: Loading and validating data...")
            df = self.load_data(raw_data_path)
            
            # Create artifact
            artifact = DataIngestionArtifact(
                raw_data_path=raw_data_path,
                total_records=len(df)
            )
            
            logger.info("="*70)
            logger.info("✅ DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"📊 Summary:")
            logger.info(f"   Location: {artifact.raw_data_path}")
            logger.info(f"   Records: {artifact.total_records:,}")
            logger.info("="*70)
            
            return artifact
        
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ DATA INGESTION FAILED")
            logger.error("="*70)
            raise SentimentAnalysisException(f"Data ingestion failed: {str(e)}", sys)
