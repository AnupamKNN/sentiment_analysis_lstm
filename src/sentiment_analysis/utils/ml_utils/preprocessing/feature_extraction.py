"""
Feature extraction utilities
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from typing import Tuple, List


class FeatureExtractor:
    """Feature extraction for NLP"""
    
    def __init__(self, max_vocab_size: int = 20000, max_sequence_length: int = 50):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.w2v_model = None
    
    def build_tokenizer(self, texts: List[str]) -> Tokenizer:
        """Build Keras tokenizer"""
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token='<OOV>',
            lower=True
        )
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        return padded
    
    def train_word2vec(self, token_lists: List[List[str]], 
                      vector_size: int = 200) -> Word2Vec:
        """Train Word2Vec embeddings"""
        self.w2v_model = Word2Vec(
            sentences=token_lists,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=10,
            seed=42
        )
        return self.w2v_model
    
    def create_embedding_matrix(self, embedding_dim: int = 200) -> np.ndarray:
        """Create embedding matrix"""
        vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_vocab_size)
        embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embedding_dim))
        
        embedding_found = 0
        for word, idx in self.tokenizer.word_index.items():
            if idx >= self.max_vocab_size:
                continue
            if word in self.w2v_model.wv:
                embedding_matrix[idx] = self.w2v_model.wv[word]
                embedding_found += 1
        
        return embedding_matrix
