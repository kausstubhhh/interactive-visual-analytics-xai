"""
Evaluation metrics for binary classification models.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(
    y_true,
    y_pred,
    y_proba
):
    """
    Calculate standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    y_proba : array-like
        Positive-class probabilities.

    Returns
    -------
    dict
        Dictionary containing:

        {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1_score": float,
            "roc_auc": float
        }
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_proba
        )
    }

    return metrics