"""
Deep learning model architectures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class AttentionLayer(layers.Layer):
    """Custom Attention Layer"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


class LSTMAttentionModel:
    """LSTM with Attention mechanism"""
    
    @staticmethod
    def build(embedding_matrix: np.ndarray,
             vocab_size: int,
             max_length: int,
             embedding_dim: int,
             lstm_units: int = 128,
             dropout_rate: float = 0.5,
             dense_units: int = 64,
             learning_rate: float = 0.001) -> models.Model:
        """
        Build LSTM + Attention model
        
        Args:
            embedding_matrix: Pre-trained embeddings
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            embedding_dim: Embedding dimensions
            lstm_units: LSTM units
            dropout_rate: Dropout rate
            dense_units: Dense layer units
            learning_rate: Learning rate
            
        Returns:
            Compiled Keras model
        """
        # Input
        input_layer = layers.Input(shape=(max_length,), name='input')
        
        # Embedding
        embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False,
            name='embedding'
        )(input_layer)
        
        # LSTM
        lstm_out = layers.LSTM(
            lstm_units,
            return_sequences=True,
            name='lstm'
        )(embedding)
        
        # Attention
        attention_out = AttentionLayer(name='attention')(lstm_out)
        
        # Dropout
        dropout1 = layers.Dropout(dropout_rate, name='dropout_1')(attention_out)
        
        # Dense
        dense = layers.Dense(dense_units, activation='relu', name='dense')(dropout1)
        dropout2 = layers.Dropout(dropout_rate * 0.6, name='dropout_2')(dense)
        
        # Output
        output = layers.Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Build model
        model = models.Model(inputs=input_layer, outputs=output, name='lstm_attention')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
