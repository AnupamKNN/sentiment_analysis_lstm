"""
Model Evaluation Component with MLflow Integration
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow import keras
import mlflow
from src.sentiment_analysis.logging.logger import logging as logger
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
from src.sentiment_analysis.entity.artifact_entity import ModelEvaluationArtifact
from src.sentiment_analysis.utils.ml_utils.metric.metric import ModelMetrics


class ModelEvaluation:
    """Handle model evaluation with MLflow tracking"""
    
    def __init__(self, model_path: str, tokenizer_path: str, use_mlflow: bool = True):
        """
        Initialize model evaluation component
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
            use_mlflow: Whether to use MLflow tracking
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.use_mlflow = use_mlflow
        
        # Create results directories
        self.results_dir = Path("training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info("Model Evaluation initialized")
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> str:
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'}
            )
            
            plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            # Add percentages
            total = cm.sum()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    percentage = (cm[i, j] / total) * 100
                    plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                            ha='center', va='center', fontsize=10, color='gray')
            
            # Save plot
            plot_path = self.plots_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Confusion matrix plot saved: {plot_path}")
            return str(plot_path)
        
        except Exception as e:
            logger.warning(f"Could not create confusion matrix plot: {str(e)}")
            return None
    
    def plot_metrics_comparison(self, metrics: dict, model_name: str) -> str:
        """
        Plot metrics bar chart
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        try:
            # Select key metrics
            key_metrics = {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc_roc']
            }
            
            # Create bar plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(key_metrics.keys(), key_metrics.values(), color='steelblue', alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontweight='bold')
            
            plt.title(f'Performance Metrics - {model_name}', fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            
            # Save plot
            plot_path = self.plots_dir / f"{model_name.replace(' ', '_')}_metrics.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Metrics plot saved: {plot_path}")
            return str(plot_path)
        
        except Exception as e:
            logger.warning(f"Could not create metrics plot: {str(e)}")
            return None
    
    def plot_roc_curve(self, y_test: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str, auc_score: float) -> str:
        """
        Plot ROC curve
        
        Args:
            y_test: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            auc_score: AUC-ROC score
            
        Returns:
            Path to saved plot
        """
        try:
            from sklearn.metrics import roc_curve
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            
            # Save plot
            plot_path = self.plots_dir / f"{model_name.replace(' ', '_')}_roc_curve.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ ROC curve plot saved: {plot_path}")
            return str(plot_path)
        
        except Exception as e:
            logger.warning(f"Could not create ROC curve: {str(e)}")
            return None
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                model_name: str = "LSTM + Attention",
                mlflow_run_id: str = None) -> ModelEvaluationArtifact:
        """
        Evaluate model and return artifact with MLflow logging
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            mlflow_run_id: MLflow run ID (if continuing existing run)
            
        Returns:
            ModelEvaluationArtifact with all evaluation results
        """
        try:
            logger.info("="*60)
            logger.info(f"EVALUATING MODEL: {model_name}")
            logger.info("="*60)
            
            # Load model
            logger.info("Loading model...")
            from src.sentiment_analysis.utils.ml_utils.model.deep_learning_models import AttentionLayer
            
            model = keras.models.load_model(
                self.model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            
            # Predict
            logger.info(f"Evaluating on {len(X_test):,} test samples...")
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = ModelMetrics.calculate_metrics(
                y_test, y_pred, y_pred_proba
            )
            
            cm = ModelMetrics.get_confusion_matrix(y_test, y_pred)
            report = ModelMetrics.get_classification_report(y_test, y_pred)
            
            # Save results to CSV
            logger.info("Saving results...")
            metrics_file = self.results_dir / f"{model_name.replace(' ', '_')}_metrics.csv"
            cm_file = self.results_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.csv"
            
            # Save metrics CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df['model_name'] = model_name
            metrics_df.to_csv(metrics_file, index=False)
            
            # Save confusion matrix CSV
            cm_df = pd.DataFrame(
                cm,
                columns=['Predicted Negative', 'Predicted Positive'],
                index=['Actual Negative', 'Actual Positive']
            )
            cm_df.to_csv(cm_file)
            
            # Create visualizations
            logger.info("Creating visualizations...")
            cm_plot_path = self.plot_confusion_matrix(cm, model_name)
            metrics_plot_path = self.plot_metrics_comparison(metrics, model_name)
            roc_plot_path = self.plot_roc_curve(y_test, y_pred_proba, model_name, metrics['auc_roc'])
            
            # Log to MLflow
            if self.use_mlflow:
                logger.info("Logging to MLflow...")
                
                # Start or continue MLflow run
                if mlflow_run_id:
                    mlflow.start_run(run_id=mlflow_run_id)
                else:
                    mlflow.start_run(run_name=f"{model_name}_evaluation")
                
                try:
                    # Log test metrics
                    mlflow.log_metrics({
                        'test_accuracy': metrics['accuracy'],
                        'test_precision': metrics['precision'],
                        'test_recall': metrics['recall'],
                        'test_f1_score': metrics['f1_score'],
                        'test_auc_roc': metrics['auc_roc']
                    })
                    
                    # Log confusion matrix values
                    mlflow.log_metrics({
                        'test_true_negatives': int(cm[0, 0]),
                        'test_false_positives': int(cm[0, 1]),
                        'test_false_negatives': int(cm[1, 0]),
                        'test_true_positives': int(cm[1, 1])
                    })
                    
                    # Log artifacts (plots and CSV files)
                    if cm_plot_path:
                        mlflow.log_artifact(cm_plot_path, "evaluation_plots")
                    if metrics_plot_path:
                        mlflow.log_artifact(metrics_plot_path, "evaluation_plots")
                    if roc_plot_path:
                        mlflow.log_artifact(roc_plot_path, "evaluation_plots")
                    
                    mlflow.log_artifact(str(metrics_file), "evaluation_results")
                    mlflow.log_artifact(str(cm_file), "evaluation_results")
                    
                    # Log classification report as text
                    report_text = str(report)
                    with open(self.results_dir / "classification_report.txt", 'w') as f:
                        f.write(report_text)
                    mlflow.log_artifact(str(self.results_dir / "classification_report.txt"), 
                                      "evaluation_results")
                    
                    logger.info("✅ Results logged to MLflow")
                
                finally:
                    if not mlflow_run_id:  # Only end if we started a new run
                        mlflow.end_run()
            
            # Determine if model is accepted
            accuracy_threshold = 0.80
            is_accepted = metrics['accuracy'] >= accuracy_threshold
            
            # Print results
            logger.info("="*60)
            logger.info("EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"✅ Performance Metrics:")
            logger.info(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall:    {metrics['recall']:.4f}")
            logger.info(f"   F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
            logger.info(f"\n📊 Confusion Matrix:")
            logger.info(f"   TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
            logger.info(f"   FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
            logger.info(f"\n🎯 Model Acceptance:")
            logger.info(f"   Threshold: {accuracy_threshold:.2f}")
            logger.info(f"   Status: {'✅ ACCEPTED' if is_accepted else '❌ REJECTED'}")
            logger.info("="*60)
            
            # Create artifact
            artifact = ModelEvaluationArtifact(
                model_name=model_name,
                accuracy=float(metrics['accuracy']),
                precision=float(metrics['precision']),
                recall=float(metrics['recall']),
                f1_score=float(metrics['f1_score']),
                auc_roc=float(metrics['auc_roc']),
                confusion_matrix=cm,
                classification_report=report,
                is_model_accepted=is_accepted,
                metrics_file_path=str(metrics_file),
                confusion_matrix_path=str(cm_file)
            )
            
            return artifact
        
        except Exception as e:
            if self.use_mlflow and not mlflow_run_id:
                mlflow.end_run()
            raise SentimentAnalysisException(f"Evaluation failed: {str(e)}", sys)
