import numpy as np
import pandas as pd


def get_local_explanation(shap_values, feature_names, instance_index):
    """
    Generate local SHAP explanation
    for a single prediction.
    """

    shap_array = shap_values.values

    # Random Forest binary classification
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    instance_values = shap_array[instance_index]

    explanation_df = pd.DataFrame(
        {"feature": feature_names, "contribution": instance_values}
    )

    explanation_df["abs_contribution"] = np.abs(explanation_df["contribution"])

    explanation_df = explanation_df.sort_values(
        by="abs_contribution", ascending=False
    ).reset_index(drop=True)

    return explanation_df
