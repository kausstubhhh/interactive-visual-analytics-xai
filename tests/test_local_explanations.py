import numpy as np

from src.explainability.local_explanations import get_local_explanation


class MockExplanation:

    def __init__(self, values):
        self.values = values


def test_local_explanation_returns_dataframe():

    shap_values = MockExplanation(np.array([[0.2, -0.1, 0.5]]))

    feature_names = ["feature_a", "feature_b", "feature_c"]

    result = get_local_explanation(shap_values, feature_names, 0)

    assert len(result) == 3


def test_local_explanation_sorted():

    shap_values = MockExplanation(np.array([[0.2, -0.1, 0.5]]))

    feature_names = ["feature_a", "feature_b", "feature_c"]

    result = get_local_explanation(shap_values, feature_names, 0)

    assert result.iloc[0]["feature"] == "feature_c"
