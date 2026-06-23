import numpy as np
import pandas as pd


def calculate_feature_importance(
    shap_values,
    feature_names
):
    """
    Calculate global SHAP feature importance.

    Parameters
    ----------
    shap_values : shap.Explanation

    feature_names : array-like

    Returns
    -------
    pandas.DataFrame
        Sorted feature importance table.
    """

    shap_array = shap_values.values

    if shap_array.ndim == 3:

        # Binary classification:
        # use positive class SHAP values

        shap_array = shap_array[:, :, 1]

    importance_scores = np.abs(
        shap_array
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


def get_top_features(
    importance_df,
    top_n=10
):
    """
    Return top N features.

    Parameters
    ----------
    importance_df : pandas.DataFrame

    top_n : int

    Returns
    -------
    pandas.DataFrame
    """

    return (
        importance_df
        .head(top_n)
        .reset_index(drop=True)
    )