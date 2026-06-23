import pandas as pd


def get_false_positives(
    y_true,
    y_pred
):
    """
    Return false positive records.
    """

    df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred
        }
    )

    mask = (
        (df["actual"] == 0)
        &
        (df["predicted"] == 1)
    )

    return df[mask]


def get_false_negatives(
    y_true,
    y_pred
):
    """
    Return false negative records.
    """

    df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred
        }
    )

    mask = (
        (df["actual"] == 1)
        &
        (df["predicted"] == 0)
    )

    return df[mask]


def calculate_error_summary(
    y_true,
    y_pred
):
    """
    Calculate error statistics.
    """

    false_positives = (
        get_false_positives(
            y_true,
            y_pred
        )
    )

    false_negatives = (
        get_false_negatives(
            y_true,
            y_pred
        )
    )

    total_errors = (
        len(false_positives)
        +
        len(false_negatives)
    )

    error_rate = (
        total_errors
        /
        len(y_true)
    )

    return {
        "false_positives":
            len(false_positives),

        "false_negatives":
            len(false_negatives),

        "total_errors":
            total_errors,

        "error_rate":
            error_rate
    }