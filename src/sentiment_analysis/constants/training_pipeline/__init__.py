"""
Training pipeline constants
"""

# Component names
DATA_INGESTION_COMPONENT = "data_ingestion"
DATA_PREPROCESSING_COMPONENT = "data_preprocessing"
FEATURE_ENGINEERING_COMPONENT = "feature_engineering"
MODEL_TRAINER_COMPONENT = "model_trainer"
MODEL_EVALUATION_COMPONENT = "model_evaluation"
MODEL_PUSHER_COMPONENT = "model_pusher"

# Training configuration
RANDOM_SEED = 42
CHUNK_SIZE = 100000
SHOW_PROGRESS = True

# Text preprocessing
LOWERCASE = True
REMOVE_URLS = True
REMOVE_MENTIONS = True
REMOVE_HASHTAGS = True