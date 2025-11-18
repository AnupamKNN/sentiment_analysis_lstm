"""
Feature Engineering Component
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import FeatureEngineeringConfig
from src.sentiment_analysis.entity.artifact_entity import (
    DataPreprocessingArtifact, FeatureEngineeringArtifact
)
from src.sentiment_analysis.utils.ml_utils.preprocessing.feature_extraction import FeatureExtractor
from src.sentiment_analysis.utils.main_utils.utils import save_object, save_numpy_array
from src.sentiment_analysis.constants.model_constants import *


class FeatureEngineering:
    """Handle feature engineering"""
    
    def __init__(self, config: FeatureEngineeringConfig,
                 preprocessing_artifact: DataPreprocessingArtifact):
        self.config = config
        self.preprocessing_artifact = preprocessing_artifact
        self.feature_extractor = FeatureExtractor(
            max_vocab_size=config.max_vocab_size,
            max_sequence_length=config.max_sequence_length
        )
        logger.info("Feature Engineering initialized")
    
    def prepare_features(self):
        """Prepare features for training - FIXED VERSION"""
        try:
            logger.info("Loading processed data...")
            df = pd.read_csv(self.preprocessing_artifact.processed_data_path)
            
            logger.info(f"Data shape before cleaning: {df.shape}")
            
            # ✅ FIX 1: Handle missing values in tokens_str
            if 'tokens_str' not in df.columns:
                raise ValueError("tokens_str column not found in processed data")
            
            # Remove rows with missing tokens_str
            initial_count = len(df)
            df = df.dropna(subset=['tokens_str'])
            df = df[df['tokens_str'].str.strip() != '']
            removed_count = initial_count - len(df)
            
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} rows with empty tokens_str")
            
            logger.info(f"Data shape after cleaning: {df.shape}")
            
            if len(df) == 0:
                raise ValueError("No valid data remaining after cleaning")
            
            # Prepare X and y
            X = df['tokens_str'].values
            y = df['sentiment'].map({0: 0, 4: 1}).values
            
            # ✅ FIX 2: Validate data types
            logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
            logger.info(f"Sample X[0]: {X[0][:100]}")
            
            # Split data
            logger.info("Splitting data...")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.15/0.85, random_state=RANDOM_SEED, stratify=y_temp
            )
            
            logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
            
            # ✅ FIX 3: Safe token reconstruction with validation
            logger.info("Reconstructing tokens...")
            
            def safe_split_tokens(x):
                """Safely split tokens string"""
                if pd.isna(x):
                    return []
                x_str = str(x).strip()
                if not x_str or x_str == 'nan':
                    return []
                return x_str.split('|')
            
            X_train_tokens = [safe_split_tokens(x) for x in X_train]
            X_val_tokens = [safe_split_tokens(x) for x in X_val]
            X_test_tokens = [safe_split_tokens(x) for x in X_test]
            
            # ✅ FIX 4: Remove empty token lists
            valid_train_indices = [i for i, tokens in enumerate(X_train_tokens) if len(tokens) > 0]
            valid_val_indices = [i for i, tokens in enumerate(X_val_tokens) if len(tokens) > 0]
            valid_test_indices = [i for i, tokens in enumerate(X_test_tokens) if len(tokens) > 0]
            
            X_train_tokens = [X_train_tokens[i] for i in valid_train_indices]
            y_train = y_train[valid_train_indices]
            
            X_val_tokens = [X_val_tokens[i] for i in valid_val_indices]
            y_val = y_val[valid_val_indices]
            
            X_test_tokens = [X_test_tokens[i] for i in valid_test_indices]
            y_test = y_test[valid_test_indices]
            
            logger.info(f"After validation - Train: {len(X_train_tokens)}, Val: {len(X_val_tokens)}, Test: {len(X_test_tokens)}")
            
            # Build tokenizer
            logger.info("Building tokenizer...")
            train_texts = [' '.join(tokens) for tokens in X_train_tokens]
            self.feature_extractor.build_tokenizer(train_texts)
            
            # Train Word2Vec
            logger.info("Training Word2Vec...")
            self.feature_extractor.train_word2vec(
                X_train_tokens, 
                vector_size=self.config.embedding_dim
            )
            
            # Create embedding matrix
            logger.info("Creating embedding matrix...")
            embedding_matrix = self.feature_extractor.create_embedding_matrix(
                embedding_dim=self.config.embedding_dim
            )
            
            # Create sequences
            logger.info("Creating sequences...")
            X_train_seq = self.feature_extractor.texts_to_sequences(train_texts)
            X_val_seq = self.feature_extractor.texts_to_sequences(
                [' '.join(tokens) for tokens in X_val_tokens]
            )
            X_test_seq = self.feature_extractor.texts_to_sequences(
                [' '.join(tokens) for tokens in X_test_tokens]
            )
            
            logger.info(f"Sequences created - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
            
            return {
                'X_train': X_train_seq, 'y_train': y_train,
                'X_val': X_val_seq, 'y_val': y_val,
                'X_test': X_test_seq, 'y_test': y_test,
                'embedding_matrix': embedding_matrix,
                'vocab_size': len(self.feature_extractor.tokenizer.word_index) + 1
            }
        
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            raise SentimentAnalysisException(f"Feature preparation failed: {str(e)}", sys)
    
    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        """Execute feature engineering"""
        try:
            logger.info("="*60)
            logger.info("Starting feature engineering")
            logger.info("="*60)
            
            # Prepare features
            data = self.prepare_features()
            
            # Create directories
            Path(self.config.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.embedding_matrix_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save artifacts
            logger.info("Saving artifacts...")
            save_object(self.config.tokenizer_path, self.feature_extractor.tokenizer)
            self.feature_extractor.w2v_model.save(self.config.word2vec_path)
            save_numpy_array(self.config.embedding_matrix_path, data['embedding_matrix'])
            
            # Save data splits
            splits_path = f"{Path(self.config.tokenizer_path).parent}/data_splits.npz"
            np.savez(
                splits_path,
                X_train=data['X_train'], y_train=data['y_train'],
                X_val=data['X_val'], y_val=data['y_val'],
                X_test=data['X_test'], y_test=data['y_test']
            )
            logger.info(f"Data splits saved: {splits_path}")
            
            artifact = FeatureEngineeringArtifact(
                tokenizer_path=self.config.tokenizer_path,
                word2vec_path=self.config.word2vec_path,
                embedding_matrix_path=self.config.embedding_matrix_path,
                vocab_size=data['vocab_size']
            )
            
            logger.info("="*60)
            logger.info("Feature engineering completed successfully")
            logger.info("="*60)
            return artifact
        
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise SentimentAnalysisException(f"Feature engineering failed: {str(e)}", sys)
