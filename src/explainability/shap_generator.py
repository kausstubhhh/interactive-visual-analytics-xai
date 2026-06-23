def generate_shap_values(
    explainer,
    X_data
):
    """
    Generate SHAP values.

    Parameters
    ----------
    explainer : shap.Explainer

    X_data : array-like

    Returns
    -------
    shap.Explanation
    """

    return explainer(
        X_data,
        check_additivity=False
    )