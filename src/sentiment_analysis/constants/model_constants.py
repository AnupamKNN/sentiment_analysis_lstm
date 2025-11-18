"""
Model-specific constants
"""

# Model architecture constants
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 200

# LSTM + Attention architecture parameters
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
DENSE_UNITS = 64

# Training configuration
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42  # ADDED - This was missing

# Early stopping
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# Word2Vec configuration
W2V_VECTOR_SIZE = 200
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 4
W2V_SG = 1  # Skip-gram
W2V_EPOCHS = 10

# Sentiment labels
SENTIMENT_NEGATIVE = 0
SENTIMENT_POSITIVE = 4

# Model filenames
MODEL_FILENAME = "lstm_attention_model.keras"
TOKENIZER_FILENAME = "tokenizer.pkl"
WORD2VEC_FILENAME = "word2vec_model.pkl"
EMBEDDING_MATRIX_FILENAME = "embedding_matrix.npy"

# Prediction thresholds
PREDICTION_THRESHOLD = 0.5
