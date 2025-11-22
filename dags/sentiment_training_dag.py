from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import sys
import os

# Ensure imports work
sys.path.append('/usr/local/airflow')

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

default_args = {
    'owner': 'anupam',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='sentiment_analysis_training_pipeline',
    default_args=default_args,
    description='End-to-end Training Pipeline',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily', 
    catchup=False,
    tags=['sentiment', 'training']
) as dag:

    @task(task_id="data_ingestion")
    def ingest():
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)
        return ingestion.initiate_data_ingestion()

    @task(task_id="data_preprocessing")
    def preprocess(ingestion_artifact):
        config = DataPreprocessingConfig()
        preprocessing = DataPreprocessing(config, ingestion_artifact)
        return preprocessing.initiate_data_preprocessing()

    @task(task_id="feature_engineering")
    def feature_eng(preprocessing_artifact):
        config = FeatureEngineeringConfig()
        fe = FeatureEngineering(config, preprocessing_artifact)
        return fe.initiate_feature_engineering()

    @task(task_id="model_training")
    def train(fe_artifact):
        config = ModelTrainerConfig()
        trainer = ModelTrainer(config, fe_artifact)
        return trainer.initiate_model_training()

    # Define the flow
    ingest_data = ingest()
    preprocessed_data = preprocess(ingest_data)
    features = feature_eng(preprocessed_data)
    trained_model = train(features)