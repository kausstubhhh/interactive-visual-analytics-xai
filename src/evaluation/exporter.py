"""
Evaluation export utilities.
"""

from pathlib import Path

import pandas as pd


def export_evaluation_summary(results, output_path):
    """
    Export evaluation metrics to CSV.
    """

    df = pd.DataFrame(results)

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return df


def export_error_summary(error_summary, output_path):
    """
    Export error summary to CSV.
    """

    df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in error_summary.items()]
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return df


def export_confusion_matrix(confusion_matrix, output_path):
    """
    Export confusion matrix to CSV.
    """

    import pandas as pd
    from pathlib import Path

    df = pd.DataFrame(
        confusion_matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)

    return df
