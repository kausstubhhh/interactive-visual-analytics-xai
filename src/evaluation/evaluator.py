"""
Unified model evaluation utilities.
"""

from sklearn.metrics import classification_report

from src.evaluation.metrics import calculate_metrics

from src.evaluation.confusion import (
    calculate_confusion_matrix,
    extract_confusion_values,
)


def evaluate_model(y_true, y_pred, y_proba):
    """
    Perform complete evaluation of a binary classifier.

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
        {
            "metrics": {...},
            "confusion_matrix": ndarray,
            "confusion_values": {...},
            "classification_report": str
        }
    """

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    confusion_matrix_result = calculate_confusion_matrix(y_true, y_pred)

    confusion_values = extract_confusion_values(y_true, y_pred)

    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_result,
        "confusion_values": confusion_values,
        "classification_report": report,
    }
