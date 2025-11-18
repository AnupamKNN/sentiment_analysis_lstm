"""
Model estimator wrapper
"""

import numpy as np
from tensorflow import keras
import pickle
from typing import Tuple

from src.sentiment_analysis.utils.ml_utils.model.deep_learning_models import AttentionLayer

class SentimentEstimator:
    """Wrapper for sentiment analysis model"""
    
    def __init__(self, model_path: str, tokenizer_path: str, max_length: int = 50):
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.max_length = max_length
    
    def predict(self, text: str) -> Tuple[int, float]:
        """
        Predict sentiment
        
        Returns:
            (predicted_label, confidence)
        """
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Preprocess
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Predict
        prob = self.model.predict(padded, verbose=0)[0][0]
        label = 1 if prob > 0.5 else 0
        confidence = prob if label == 1 else (1 - prob)
        
        return label, float(confidence)