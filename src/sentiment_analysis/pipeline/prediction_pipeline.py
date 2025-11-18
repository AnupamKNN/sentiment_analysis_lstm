"""
Prediction Pipeline
"""

import sys
import time
import numpy as np
from tensorflow import keras
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.config_entity import PredictionPipelineConfig
from src.sentiment_analysis.entity.artifact_entity import PredictionArtifact
from src.sentiment_analysis.utils.ml_utils.preprocessing import TextPreprocessor
from src.sentiment_analysis.utils.main_utils.utils import load_object
from src.sentiment_analysis.constants.prediction_pipeline import OUTPUT_LABELS


class PredictionPipeline:
    """Single text prediction pipeline"""
    
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.model = None
        self.tokenizer = None
        logger.info("Prediction Pipeline initialized")
    
    def load_model(self):
        """Load trained model and tokenizer"""
        try:
            if self.model is None:
                logger.info("Loading model...")
                from src.sentiment_analysis.utils.ml_utils.model.deep_learning_models import AttentionLayer
                
                self.model = keras.models.load_model(
                    self.config.model_path,
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
                logger.info("Model loaded successfully")
            
            if self.tokenizer is None:
                logger.info("Loading tokenizer...")
                self.tokenizer = load_object(self.config.tokenizer_path)
                logger.info("Tokenizer loaded successfully")
        
        except Exception as e:
            raise SentimentAnalysisException(f"Model loading failed: {str(e)}", sys)
    
    def preprocess_text(self, text: str) -> tuple:
        """Preprocess input text"""
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Clean text
            cleaned_text = TextPreprocessor.clean_social_media_text(text)
            
            # Convert to sequence
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(
                sequence,
                maxlen=self.config.max_sequence_length,
                padding='post',
                truncating='post'
            )
            
            return padded, cleaned_text
        
        except Exception as e:
            raise SentimentAnalysisException(f"Preprocessing failed: {str(e)}", sys)
    
    def predict(self, text: str) -> PredictionArtifact:
        """Predict sentiment for single text"""
        try:
            start_time = time.time()
            
            # Load model if not loaded
            self.load_model()
            
            # Preprocess
            padded_sequence, cleaned_text = self.preprocess_text(text)
            
            # Predict
            prediction_proba = self.model.predict(padded_sequence, verbose=0)[0][0]
            predicted_label = 1 if prediction_proba > self.config.prediction_threshold else 0
            
            # Calculate confidence
            confidence = prediction_proba if predicted_label == 1 else (1 - prediction_proba)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create artifact
            artifact = PredictionArtifact(
                text=text,
                cleaned_text=cleaned_text,
                predicted_label=predicted_label,
                predicted_sentiment=OUTPUT_LABELS[predicted_label],
                confidence=float(confidence),
                prediction_probability=float(prediction_proba),
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"Prediction: {artifact.predicted_sentiment} ({confidence:.2%})")
            
            return artifact
        
        except Exception as e:
            raise SentimentAnalysisException(f"Prediction failed: {str(e)}", sys)
