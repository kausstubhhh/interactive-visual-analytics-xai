"""
Evaluation service layer.

Coordinates model evaluation
and export generation.
"""

from pathlib import Path

from src.evaluation.evaluator import (
    evaluate_model
)

from src.evaluation.misclassification import (
    calculate_error_summary
)

from src.evaluation.exporter import (
    export_evaluation_summary,
    export_error_summary
)

from src.evaluation.exporter import (
    export_evaluation_summary,
    export_error_summary,
    export_confusion_matrix
)

EXPORT_DIR = (
    Path("data")
    / "exports"
)


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba
):
    """
    Evaluate a model.
    """

    return evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba
    )


def generate_error_analysis(
    y_true,
    y_pred
):
    """
    Generate error statistics.
    """

    return calculate_error_summary(
        y_true,
        y_pred
    )


def export_results(
    results
):
    """
    Export evaluation summary.
    """

    output_file = (
        EXPORT_DIR
        / "evaluation_summary.csv"
    )

    return export_evaluation_summary(
        results=results,
        output_path=output_file
    )


def evaluate_all_models(
    dataset_name,
    y_true,
    model_results
):
    """
    Evaluate all trained models.
    """

    evaluation_rows = []

    error_results = {}

    for (
        model_name,
        result
    ) in model_results.items():

        evaluation = (
            evaluate_predictions(
                y_true=y_true,
                y_pred=result[
                    "predictions"
                ],
                y_proba=result[
                    "probabilities"
                ][:, 1]
            )
        )

        confusion_file = (
            EXPORT_DIR
            / "confusion"
            / f"{dataset_name}_{model_name}_confusion.csv"
        )

        export_confusion_matrix(
            confusion_matrix=evaluation[
                "confusion_matrix"
            ],
            output_path=confusion_file
        )
        errors = (
            generate_error_analysis(
                y_true=y_true,
                y_pred=result[
                    "predictions"
                ]
            )
        )

        metrics = (
            evaluation[
                "metrics"
            ]
        )

        evaluation_rows.append(
            {
                "dataset":
                    dataset_name,

                "model":
                    model_name,

                **metrics
            }
        )

        error_results[
            model_name
        ] = errors

        error_file = (
            EXPORT_DIR
            / "errors"
            / f"{dataset_name}_{model_name}_errors.csv"
        )

        export_error_summary(
            error_summary=errors,
            output_path=error_file
        )

    export_results(
        evaluation_rows
    )

    return {
        "evaluation":
            evaluation_rows,

        "errors":
            error_results
    }