from pathlib import Path

from src.data.loader import load_dataset
from src.data.schema_detector import detect_schema
from src.data.preprocess import prepare_dataset

import numpy as np

from src.explainability.importance import (
    calculate_feature_importance
)


class MockShapValues:

    def __init__(self, values):
        self.values = values


def test_feature_importance():

    shap_values = MockShapValues(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0]
            ]
        )
    )

    feature_names = [
        "feature_a",
        "feature_b"
    ]

    results = (
        calculate_feature_importance(
            shap_values,
            feature_names
        )
    )

    assert len(results) == 2

    assert (
        "feature"
        in results.columns
    )

    assert (
        "importance"
        in results.columns
    )

from src.models.logistic_regression import (
    build_logistic_regression
)

from src.models.trainer import (
    fit_and_predict
)

from src.explainability.shap_explainer import (
    build_shap_explainer
)

from src.explainability.shap_generator import (
    generate_shap_values
)


DATASET_PATH = (
    Path("data/raw/heloc_dataset_v1.csv")
)

TARGET_COLUMN = (
    "RiskPerformance"
)


def main():

    print("\nLoading dataset...")

    df = load_dataset(
        DATASET_PATH
    )

    schema = detect_schema(
        df,
        TARGET_COLUMN
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_dataset(
        df,
        TARGET_COLUMN,
        schema
    )

    print("Preprocessing data...")

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print("Training Logistic Regression...")

    model = (
        build_logistic_regression()
    )

    fit_and_predict(
        model,
        X_train_processed,
        y_train,
        X_test_processed
    )

    print("Building SHAP explainer...")

    explainer = (
        build_shap_explainer(
            model,
            X_train_processed
        )
    )

    print("Generating SHAP values...")

    shap_values = (
        generate_shap_values(
            explainer,
            X_test_processed
        )
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importance_df = (
    calculate_feature_importance(
        shap_values,
        feature_names
        )
    )

    print("\nTop 10 Features\n")

    print(
        importance_df.head(10)
    )
    
    print("\n" + "=" * 60)
    print("SHAP VALIDATION")
    print("=" * 60)

    print(
        f"SHAP shape: "
        f"{shap_values.values.shape}"
    )

    print(
        f"Feature count: "
        f"{len(feature_names)}"
    )

    print("\nFirst 10 feature names:\n")

    for feature in feature_names[:10]:
        print(feature)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()