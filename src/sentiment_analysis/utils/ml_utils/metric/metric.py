"""
Model evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple


class ModelMetrics:
    """Calculate model performance metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_pred_proba: np.ndarray = None) -> Dict:
        """Calculate all metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  target_names: list = None) -> Dict:
        """Get classification report"""
        if target_names is None:
            target_names = ['Negative', 'Positive']
        
        return classification_report(y_true, y_pred, 
                                    target_names=target_names,
                                    output_dict=True)