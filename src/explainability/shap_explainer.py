import shap


def build_shap_explainer(
    model,
    background_data
):
    """
    Create SHAP explainer.
    """

    return shap.Explainer(
        model,
        background_data
    )