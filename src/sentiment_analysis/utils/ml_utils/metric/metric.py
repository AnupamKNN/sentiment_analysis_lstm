"""
Model evaluation metrics for classification
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from src.sentiment_analysis.exception.exception import SentimentAnalysisException
import numpy as np
import sys
from pathlib import Path
import os
import pandas as pd
from datetime import datetime

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
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = None
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                auc = None

        # confusion matrix: tn, fp, fn, tp for binary classification
        try:
            cm = confusion_matrix(y_true, y_pred)
            # ensure it's 2x2
            if cm.shape == (2, 2):
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
                cm_list = [[tn, fp], [fn, tp]]
            else:
                # for multi-class or unexpected shapes, store raw cm
                tn = fp = fn = tp = None
                cm_list = cm.tolist()
        except Exception:
            tn = fp = fn = tp = None
            cm_list = None

        result = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'auc_roc': float(auc) if auc is not None else None,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'confusion_matrix': cm_list
        }

        return result
    except Exception as e:
        raise SentimentAnalysisException(e, sys)


def save_metrics(metrics: dict, model_name: str = 'model', output_dir: str = None) -> str:
    """Save metrics dict to a CSV file under `training_results/` (or provided `output_dir`).

    The CSV file will be named `<model_name>_metrics.csv` and will contain a single-row snapshot
    with a timestamp column `timestamp` so multiple runs can be appended later if desired.

    Returns the path to the written CSV file.
    """
    try:
        repo_root = Path.cwd()
        training_dir = Path(output_dir) if output_dir else Path(os.getenv('TRAINING_RESULTS_DIR', repo_root / 'training_results'))
        training_dir.mkdir(parents=True, exist_ok=True)

        filename = training_dir / f"{model_name}_metrics.csv"

        # Prepare row data: ensure deterministic columns order
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': model_name,
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'auc_roc': metrics.get('auc_roc'),
            'tn': metrics.get('tn'),
            'fp': metrics.get('fp'),
            'fn': metrics.get('fn'),
            'tp': metrics.get('tp'),
            'confusion_matrix': metrics.get('confusion_matrix')
        }

        # If file exists, append; otherwise create with header
        if filename.exists():
            # append a new row
            df = pd.DataFrame([row])
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df = pd.DataFrame([row])
            df.to_csv(filename, index=False)

        # also write model metadata for quick access by the API/frontend
        try:
            metadata = {
                "model_name": model_name,
                "last_trained": row['timestamp'],
                "model_version": datetime.utcnow().isoformat()
            }
            meta_file = training_dir / "model_metadata.json"
            pd.Series(metadata).to_json(meta_file, orient='index')
        except Exception:
            # non-fatal: ignore metadata write failures
            pass

        return str(filename)
    except Exception as e:
        raise SentimentAnalysisException(e, sys)
