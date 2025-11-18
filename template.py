import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "sentiment_analysis"

# Complete list of files with updated 3-notebook structure
list_of_files = [
    # CI/CD and GitHub Actions
    ".github/workflows/ci.yaml",
    ".github/workflows/cd.yaml",
    ".github/workflows/docker-build.yaml",
    
    # Data schema and configuration
    f"data_schema/schema.yaml",
    f"data_schema/preprocessing_config.yaml",
    f"data_schema/model_config.yaml",
    
    # Raw project data
    f"project_data/raw/sentiment140.csv",
    f"project_data/processed/train.csv",
    f"project_data/processed/test.csv",
    f"project_data/processed/validation.csv",
    f"project_data/sample.txt",
    
    # Research notebooks - UPDATED 3-notebook structure
    f"research_notebooks/01_data_exploration_preprocessing_comparison.ipynb",
    f"research_notebooks/02_feature_engineering_comparison.ipynb", 
    f"research_notebooks/03_model_development_evaluation_optimization.ipynb",
    f"research_notebooks/data/sample_data.txt",
    f"research_notebooks/exports/model_comparison_results.csv",
    f"research_notebooks/exports/best_model_metrics.json",
    f"research_notebooks/exports/preprocessing_results.csv",
    f"research_notebooks/exports/feature_engineering_results.csv",
    
    # Core source code structure
    f"src/__init__.py",
    f"src/{project_name}/__init__.py",
    
    # Components - modular ML pipeline
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/feature_engineering.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    
    # Utilities
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils/__init__.py",
    f"src/{project_name}/utils/main_utils/utils.py",
    
    # ML utilities with NLP specific modules
    f"src/{project_name}/utils/ml_utils/__init__.py",
    f"src/{project_name}/utils/ml_utils/model/__init__.py",
    f"src/{project_name}/utils/ml_utils/model/estimator.py",
    f"src/{project_name}/utils/ml_utils/model/nlp_models.py",
    f"src/{project_name}/utils/ml_utils/model/deep_learning_models.py",
    f"src/{project_name}/utils/ml_utils/metric/__init__.py",
    f"src/{project_name}/utils/ml_utils/metric/metric.py",
    f"src/{project_name}/utils/ml_utils/preprocessing/__init__.py",
    f"src/{project_name}/utils/ml_utils/preprocessing/text_preprocessing.py",
    f"src/{project_name}/utils/ml_utils/preprocessing/feature_extraction.py",
    
    # Configuration management
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    
    # Pipeline orchestration
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/pipeline/batch_prediction_pipeline.py",
    
    # Entity definitions
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    
    # Exception handling
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/exception.py",
    
    # Logging system
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/logging/logger.py",
    
    # Constants and configurations
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/constants/training_pipeline/__init__.py",
    f"src/{project_name}/constants/prediction_pipeline/__init__.py",
    f"src/{project_name}/constants/model_constants.py",
    
    # MLOps and monitoring
    f"src/{project_name}/monitoring/__init__.py",
    f"src/{project_name}/monitoring/model_monitoring.py",
    f"src/{project_name}/monitoring/data_drift_detection.py",
    
    # API and web application
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/routes.py",
    f"src/{project_name}/api/schemas.py",
    
    # Model artifacts and storage
    f"artifacts/models/.gitkeep",
    f"artifacts/data/.gitkeep",
    f"artifacts/reports/.gitkeep",
    f"artifacts/experiments/.gitkeep",
    
    # Models directory for saved models
    f"models/preprocessing/.gitkeep",
    f"models/trained_models/.gitkeep",
    f"models/vectorizers/.gitkeep",
    
    # Airflow DAGs for workflow orchestration
    f"airflow_dags/__init__.py",
    f"airflow_dags/sentiment_analysis_pipeline.py",
    f"airflow_dags/data_ingestion_dag.py",
    f"airflow_dags/model_training_dag.py",
    
    # Configuration files
    f"config/config.yaml",
    f"config/model_config.yaml",
    f"config/data_config.yaml",
    f"config/deployment_config.yaml",
    
    # Testing framework
    f"tests/__init__.py",
    f"tests/unit/__init__.py",
    f"tests/unit/test_preprocessing.py",
    f"tests/unit/test_models.py",
    f"tests/integration/__init__.py",
    f"tests/integration/test_pipeline.py",
    
    # Documentation
    f"docs/README.md",
    f"docs/api_documentation.md",
    f"docs/model_documentation.md",
    f"docs/deployment_guide.md",
    
    # Scripts for automation
    f"scripts/download_data.py",
    f"scripts/setup_environment.py",
    f"scripts/train_model.py",
    f"scripts/evaluate_model.py",
    
    # Root level files
    "main.py",
    "app.py",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    ".dockerignore",
    "setup.py",
    "requirements.txt",
    "README.md",
    "LICENSE",
    ".env.example",
    "pyproject.toml",
    
    # Templates for web interface
    f"templates/index.html",
    f"templates/results.html",
    f"templates/dashboard.html",
    
    # Static files for web interface
    f"static/css/style.css",
    f"static/js/main.js",
    f"static/images/.gitkeep",
    
    # MLflow and experiment tracking
    f"mlruns/.gitkeep",
    f"experiments/mlflow_experiments.py",
    f"experiments/hyperparameter_tuning.py",
]

print("="*80)
print("CREATING FOLDER STRUCTURE FOR VELOCISENSE ANALYTICS")
print("SOCIAL MEDIA SENTIMENT & TREND ANALYSIS PLATFORM")
print("="*80)

# Create the folder structure
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

print("\n" + "="*80)
print("✅ FOLDER STRUCTURE CREATED SUCCESSFULLY!")
print("="*80)
