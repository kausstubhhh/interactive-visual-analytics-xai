"""
Confusion matrix utilities for binary classification.
"""

from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(
    y_true,
    y_pred
):
    """
    Calculate confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    ndarray
        [[TN, FP],
         [FN, TP]]
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def extract_confusion_values(
    y_true,
    y_pred
):
    """
    Extract TN, FP, FN and TP.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        {
            "tn": int,
            "fp": int,
            "fn": int,
            "tp": int
        }
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }