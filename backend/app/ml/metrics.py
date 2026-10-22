"""Metrics and evaluation for fraud detection models."""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FraudDetectionMetrics:
    """Calculate fraud detection metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_class: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:
        """Calculate comprehensive metrics for fraud detection.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            y_pred_class: Predicted classes
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred_class)),
            "precision": float(precision_score(y_true, y_pred_class, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_class, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_class, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_class)
        metrics["confusion_matrix"] = cm.tolist()

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        metrics["class_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}

        # Fraud-specific metrics
        if len(cm) == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["specificity"] = float(specificity)

            # Fraud detection rate
            metrics["fraud_detection_rate"] = metrics["recall"]

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

        return metrics

    @staticmethod
    def compare_models(metrics_baseline: Dict, metrics_protected: Dict) -> Dict:
        """Compare baseline and protected models.
        
        Args:
            metrics_baseline: Baseline model metrics
            metrics_protected: Protected model metrics
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "baseline": metrics_baseline,
            "protected": metrics_protected,
            "differences": {},
        }

        # Calculate differences
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if key in metrics_baseline and key in metrics_protected:
                diff = metrics_protected[key] - metrics_baseline[key]
                pct_change = (diff / abs(metrics_baseline[key]) * 100) if metrics_baseline[key] != 0 else 0
                comparison["differences"][key] = {
                    "value": float(diff),
                    "percent": float(pct_change),
                }

        return comparison
