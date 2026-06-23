from pathlib import Path

import pandas as pd

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

from src.evaluation.misclassification import (
    calculate_error_summary
)


DATA_DIR = Path("data/raw")

EXPORT_DIR = Path(
    "data/exports/errors"
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


def export_error_summary(
    summary,
    output_file
):

    df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value
            }
            for key, value
            in summary.items()
        ]
    )

    df.to_csv(
        output_file,
        index=False
    )


def run_analysis(
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

    for (
        model_name,
        model_builder
    ) in MODELS.items():

        print(
            f"\nAnalysing {model_name}"
        )

        model = model_builder()

        results = fit_and_predict(
            model,
            X_train_processed,
            y_train,
            X_test_processed
        )

        summary = (
            calculate_error_summary(
                y_test,
                results["predictions"]
            )
        )

        output_file = (
            EXPORT_DIR
            /
            f"{dataset_name.lower()}_"
            f"{model_name}_errors.csv"
        )

        export_error_summary(
            summary,
            output_file
        )

        print(
            f"Exported: {output_file}"
        )

        print(summary)


if __name__ == "__main__":

    for (
        dataset_name,
        config
    ) in DATASETS.items():

        run_analysis(
            dataset_name,
            config["path"],
            config["target"]
        )