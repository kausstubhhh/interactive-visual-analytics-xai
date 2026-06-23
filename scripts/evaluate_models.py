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

from src.evaluation.evaluator import (
    evaluate_model
)

from src.evaluation.exporter import (
    export_evaluation_summary
)


DATA_DIR = Path("data/raw")


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


def print_results(
    dataset_name,
    model_name,
    results
):
    """
    Print evaluation results to console.
    """

    print("\n" + "=" * 60)
    print(dataset_name)
    print(model_name)
    print("=" * 60)

    metrics = results["metrics"]

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC AUC  : {metrics['roc_auc']:.4f}"
    )

    confusion = results[
        "confusion_values"
    ]

    print("\nConfusion Matrix Values")

    print(
        f"TN: {confusion['tn']}"
    )

    print(
        f"FP: {confusion['fp']}"
    )

    print(
        f"FN: {confusion['fn']}"
    )

    print(
        f"TP: {confusion['tp']}"
    )


def evaluate_dataset(
    dataset_name,
    dataset_path,
    target_column,
    summary_results
):
    """
    Train and evaluate all models
    for a dataset.
    """

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

    models = {
        "Logistic Regression":
            build_logistic_regression(),

        "Random Forest":
            build_random_forest()
    }

    for (
        model_name,
        model
    ) in models.items():

        prediction_results = (
            fit_and_predict(
                model,
                X_train_processed,
                y_train,
                X_test_processed
            )
        )

        evaluation_results = (
            evaluate_model(
                y_true=y_test,
                y_pred=prediction_results[
                    "predictions"
                ],
                y_proba=prediction_results[
                    "probabilities"
                ][:, 1]
            )
        )

        print_results(
            dataset_name,
            model_name,
            evaluation_results
        )

        metrics = (
            evaluation_results[
                "metrics"
            ]
        )

        summary_results.append(
            {
                "dataset":
                    dataset_name,

                "model":
                    model_name,

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1_score":
                    metrics["f1_score"],

                "roc_auc":
                    metrics["roc_auc"]
            }
        )


if __name__ == "__main__":

    summary_results = []

    for (
        dataset_name,
        config
    ) in DATASETS.items():

        evaluate_dataset(
            dataset_name,
            config["path"],
            config["target"],
            summary_results
        )

    export_evaluation_summary(
        results=summary_results,
        output_path=
        "data/exports/evaluation_summary.csv"
    )

    print(
        "\nEvaluation summary exported."
    )