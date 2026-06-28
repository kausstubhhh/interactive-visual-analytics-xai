import numpy as np

from src.explainability.importance import calculate_feature_importance, get_top_features


class MockShapValues:

    def __init__(self, values):
        self.values = values


def test_feature_importance_returns_dataframe():

    shap_values = MockShapValues(np.array([[1.0, 2.0], [3.0, 4.0]]))

    feature_names = ["feature_a", "feature_b"]

    results = calculate_feature_importance(shap_values, feature_names)

    assert len(results) == 2

    assert "feature" in results.columns
    assert "importance" in results.columns


def test_get_top_features():

    shap_values = MockShapValues(np.array([[1.0, 2.0], [3.0, 4.0]]))

    feature_names = ["feature_a", "feature_b"]

    results = calculate_feature_importance(shap_values, feature_names)

    top_features = get_top_features(results, top_n=1)

    assert len(top_features) == 1
