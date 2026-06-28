import pandas as pd

from src.explainability.exporter import export_feature_importance


def test_export_feature_importance(tmp_path):

    importance_df = pd.DataFrame({"feature": ["feature_a"], "importance": [0.5]})

    output_file = tmp_path / "importance.csv"

    export_feature_importance(importance_df, output_file)

    assert output_file.exists()
