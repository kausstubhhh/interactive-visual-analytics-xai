import pandas as pd
import numpy as np


def calculate_feature_importance(
    shap_values,
    feature_names
):
    """
    Calculate global SHAP feature importance.

    Parameters
    ----------
    shap_values : shap.Explanation

    feature_names : list-like

    Returns
    -------
    pandas.DataFrame
    """

    importance_scores = np.abs(
        shap_values.values
    ).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_scores
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df