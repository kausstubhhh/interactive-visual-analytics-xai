"""
Generic model training utilities.
"""


def train_model(model, X_train, y_train):
    """
    Train a classifier.

    Parameters
    ----------
    model : sklearn estimator

    X_train : array-like

    y_train : array-like

    Returns
    -------
    fitted model
    """

    model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    """
    Generate class predictions.
    """

    return model.predict(X_test)


def predict_proba(model, X_test):
    """
    Generate prediction probabilities.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)

    raise AttributeError(f"{type(model).__name__} does not support predict_proba()")


def fit_and_predict(model, X_train, y_train, X_test):
    """
    Train model and generate outputs.

    Returns
    -------
    dict
    """

    trained_model = train_model(model, X_train, y_train)

    predictions = predict(trained_model, X_test)

    probabilities = predict_proba(trained_model, X_test)

    return {
        "model": trained_model,
        "predictions": predictions,
        "probabilities": probabilities,
    }
