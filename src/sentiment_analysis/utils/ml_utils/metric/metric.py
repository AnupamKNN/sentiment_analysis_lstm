"""
Model evaluation metrics for classification
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
import numpy as np
import sys

def get_classification_score(y_true, y_pred, y_pred_proba=None):
    """
    Calculate main classification metrics.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities for positive class (optional)

    Returns:
        dict: metrics including accuracy, precision, recall, f1_score, and optionally auc_roc
    """
    try:
        result = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
        }
        if y_pred_proba is not None:
            result['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            result['auc_roc'] = None
        return result
    except Exception as e:
        raise SentimentAnalysisException(e, sys)
