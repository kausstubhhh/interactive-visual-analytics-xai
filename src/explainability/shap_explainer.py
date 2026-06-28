import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_shap_explainer(
    model,
    background_data
):
    """
    Create the appropriate SHAP explainer.
    """

    if isinstance(
        model,
        RandomForestClassifier
    ):
        return shap.TreeExplainer(model)

    if isinstance(
        model,
        LogisticRegression
    ):
        return shap.LinearExplainer(
            model,
            background_data
        )

    return shap.Explainer(
        model,
        background_data
    )