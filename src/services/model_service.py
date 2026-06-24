"""
Model service layer.

Coordinates model creation
and training.
"""

from src.models.logistic_regression import (
    build_logistic_regression
)

from src.models.random_forest import (
    build_random_forest
)

from src.models.trainer import (
    fit_and_predict
)


def train_logistic_regression(
    X_train,
    y_train,
    X_test
):
    """
    Train Logistic Regression.
    """

    model = build_logistic_regression()

    return fit_and_predict(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test
    )


def train_random_forest(
    X_train,
    y_train,
    X_test
):
    """
    Train Random Forest.
    """

    model = build_random_forest()

    return fit_and_predict(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test
    )


def train_models(
    X_train,
    y_train,
    X_test
):
    """
    Train all project models.
    """

    logistic_results = (
        train_logistic_regression(
            X_train,
            y_train,
            X_test
        )
    )

    random_forest_results = (
        train_random_forest(
            X_train,
            y_train,
            X_test
        )
    )

    return {
        "logistic_regression":
            logistic_results,

        "random_forest":
            random_forest_results
    }