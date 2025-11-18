"""
Training Pipeline
"""

import sys
import numpy as np
from pathlib import Path
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    FeatureEngineeringConfig,
    ModelTrainerConfig
)
from src.sentiment_analysis.components.data_ingestion import DataIngestion
from src.sentiment_analysis.components.data_preprocessing import DataPreprocessing
from src.sentiment_analysis.components.feature_engineering import FeatureEngineering
from src.sentiment_analysis.components.model_trainer import ModelTrainer
from src.sentiment_analysis.components.model_evaluation import ModelEvaluation  # ADDED
from src.sentiment_analysis.components.model_pusher import ModelPusher  # ADDED


class TrainingPipeline:
    """Complete training pipeline with evaluation"""
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.feature_engineering_config = FeatureEngineeringConfig()
        self.model_trainer_config = ModelTrainerConfig()
        
        logger.info("Training Pipeline initialized")
    
    def start_data_ingestion(self):
        """Start data ingestion"""
        try:
            logger.info("="*50)
            logger.info("STEP 1: DATA INGESTION")
            logger.info("="*50)
            
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logger.info("✅ Data ingestion completed")
            return data_ingestion_artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Data ingestion failed: {str(e)}", sys)
    
    def start_data_preprocessing(self, data_ingestion_artifact):
        """Start data preprocessing"""
        try:
            logger.info("="*50)
            logger.info("STEP 2: DATA PREPROCESSING")
            logger.info("="*50)
            
            data_preprocessing = DataPreprocessing(
                self.data_preprocessing_config,
                data_ingestion_artifact
            )
            data_preprocessing_artifact = data_preprocessing.initiate_data_preprocessing()
            
            logger.info("✅ Data preprocessing completed")
            return data_preprocessing_artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Data preprocessing failed: {str(e)}", sys)
    
    def start_feature_engineering(self, data_preprocessing_artifact):
        """Start feature engineering"""
        try:
            logger.info("="*50)
            logger.info("STEP 3: FEATURE ENGINEERING")
            logger.info("="*50)
            
            feature_engineering = FeatureEngineering(
                self.feature_engineering_config,
                data_preprocessing_artifact
            )
            feature_engineering_artifact = feature_engineering.initiate_feature_engineering()
            
            logger.info("✅ Feature engineering completed")
            return feature_engineering_artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Feature engineering failed: {str(e)}", sys)
    
    def start_model_training(self, feature_engineering_artifact):
        """Start model training"""
        try:
            logger.info("="*50)
            logger.info("STEP 4: MODEL TRAINING")
            logger.info("="*50)
            
            model_trainer = ModelTrainer(
                self.model_trainer_config,
                feature_engineering_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_training()
            
            logger.info("✅ Model training completed")
            return model_trainer_artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Model training failed: {str(e)}", sys)
    
    def start_model_evaluation(self, model_trainer_artifact, feature_artifact):
        """Evaluate trained model on test set"""
        try:
            logger.info("="*50)
            logger.info("STEP 5: MODEL EVALUATION")
            logger.info("="*50)
            
            # Load test data
            data_path = f"{Path(feature_artifact.tokenizer_path).parent}/data_splits.npz"
            data = np.load(data_path)
            X_test, y_test = data['X_test'], data['y_test']
            
            logger.info(f"Evaluating on {len(X_test):,} test samples")
            
            # Evaluate model
            model_evaluation = ModelEvaluation(
                model_path=model_trainer_artifact.trained_model_path,
                tokenizer_path=feature_artifact.tokenizer_path
            )
            
            # Returns ModelEvaluationArtifact now
            evaluation_artifact = model_evaluation.evaluate(
                X_test, y_test, 
                model_name="LSTM + Attention"
            )
            
            logger.info("✅ Model evaluation completed")
            return evaluation_artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Model evaluation failed: {str(e)}", sys)

    
    def start_model_pusher(self, model_trainer_artifact, evaluation_results):
        """
        NEW: Push model to production only if it meets criteria
        """
        try:
            logger.info("="*50)
            logger.info("STEP 6: MODEL PUSHER")
            logger.info("="*50)
            
            # Check if model meets minimum criteria
            accuracy_threshold = 0.80  # 80% minimum accuracy
            metrics = evaluation_results['metrics']
            
            if metrics['accuracy'] >= accuracy_threshold:
                logger.info(f"✅ Model meets criteria (accuracy: {metrics['accuracy']:.4f} >= {accuracy_threshold})")
                
                # Push model to production
                model_pusher = ModelPusher(
                    model_path=model_trainer_artifact.trained_model_path,
                    final_path=model_trainer_artifact.trained_model_path  # Already in final_models
                )
                
                pusher_artifact = model_pusher.push_model()
                
                logger.info("✅ Model pushed to production")
                return pusher_artifact
            else:
                logger.warning(f"⚠️  Model does not meet criteria (accuracy: {metrics['accuracy']:.4f} < {accuracy_threshold})")
                logger.warning("Model NOT pushed to production")
                return None
        
        except Exception as e:
            raise SentimentAnalysisException(f"Model pusher failed: {str(e)}", sys)
    
    def run_pipeline(self):
        """Execute complete training pipeline"""
        try:
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            # Returns: DataIngestionArtifact
            
            # Step 2: Data Preprocessing
            data_preprocessing_artifact = self.start_data_preprocessing(
                data_ingestion_artifact
            )
            # Returns: DataPreprocessingArtifact
            
            # Step 3: Feature Engineering
            feature_engineering_artifact = self.start_feature_engineering(
                data_preprocessing_artifact
            )
            # Returns: FeatureEngineeringArtifact
            
            # Step 4: Model Training
            model_trainer_artifact = self.start_model_training(
                feature_engineering_artifact
            )
            # Returns: ModelTrainerArtifact
            
            # Step 5: Model Evaluation
            evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact,
                feature_engineering_artifact
            )
            # Returns: ModelEvaluationArtifact
            
            # Step 6: Model Pusher
            pusher_artifact = self.start_model_pusher(
                model_trainer_artifact,
                evaluation_artifact
            )
            # Returns: ModelPusherArtifact
            
            logger.info("✅ TRAINING PIPELINE COMPLETED")
            
            return {
                'data_ingestion': data_ingestion_artifact,
                'preprocessing': data_preprocessing_artifact,
                'feature_engineering': feature_engineering_artifact,
                'model_trainer': model_trainer_artifact,
                'evaluation': evaluation_artifact,
                'pusher': pusher_artifact
            }
        
        except Exception as e:
            raise SentimentAnalysisException(f"Pipeline failed: {str(e)}", sys)

