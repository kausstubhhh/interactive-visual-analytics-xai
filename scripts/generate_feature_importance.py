from pathlib import Path

from src.data.loader import load_dataset
from src.data.schema_detector import detect_schema
from src.data.preprocess import prepare_dataset

from src.models.logistic_regression import (
    build_logistic_regression
)

from src.models.random_forest import (
    build_random_forest
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

from src.explainability.importance import (
    calculate_feature_importance
)

from src.explainability.exporter import (
    export_feature_importance
)


DATA_DIR = Path("data/raw")

OUTPUT_DIR = Path(
    "data/exports/shap"
)

# Use a representative sample for SHAP
SHAP_SAMPLE_SIZE = 500


DATASETS = {
    "HELOC": {
        "path": DATA_DIR / "heloc_dataset_v1.csv",
        "target": "RiskPerformance"
    },
    "BANK": {
        "path": DATA_DIR / "bank-additional-full.xlsx",
        "target": "y"
    }
}


MODELS = {
    "logistic_regression":
        build_logistic_regression,

    "random_forest":
        build_random_forest
}


def generate_importance_for_model(
    dataset_name,
    dataset_path,
    target_column
):

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    df = load_dataset(
        dataset_path
    )

    schema = detect_schema(
        df,
        target_column
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_dataset(
        df,
        target_column,
        schema
    )

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

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Sample data for SHAP
    shap_rows = min(
        SHAP_SAMPLE_SIZE,
        X_test_processed.shape[0]
    )

    X_shap = (
        X_test_processed[:shap_rows]
    )

    print(
        f"\nUsing {shap_rows} rows "
        f"for SHAP analysis"
    )

    for (
        model_name,
        model_builder
    ) in MODELS.items():

        print(
            f"\nGenerating SHAP for "
            f"{model_name}"
        )

        model = model_builder()

        fit_and_predict(
            model,
            X_train_processed,
            y_train,
            X_test_processed
        )

        explainer = (
            build_shap_explainer(
                model,
                X_train_processed
            )
        )

        shap_values = (
            generate_shap_values(
                explainer,
                X_shap
            )
        )

        importance_df = (
            calculate_feature_importance(
                shap_values,
                feature_names
            )
        )

        filename = (
            f"{dataset_name.lower()}_"
            f"{model_name}_"
            f"importance.csv"
        )

        output_file = (
            OUTPUT_DIR /
            filename
        )

        export_feature_importance(
            importance_df,
            output_file
        )

        print(
            f"Exported: "
            f"{output_file}"
        )

        print(
            "\nTop 5 Features"
        )

        print(
            importance_df
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":

    for (
        dataset_name,
        config
    ) in DATASETS.items():

        generate_importance_for_model(
            dataset_name,
            config["path"],
            config["target"]
        )