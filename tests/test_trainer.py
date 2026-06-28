from src.models.logistic_regression import build_logistic_regression

from src.models.trainer import train_model, predict, predict_proba, fit_and_predict

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def create_sample_data():

    X, y = load_breast_cancer(return_X_y=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_train_model():

    X_train, X_test, y_train, y_test = create_sample_data()

    model = build_logistic_regression()

    trained = train_model(model, X_train, y_train)

    assert trained is not None


def test_predict():

    X_train, X_test, y_train, y_test = create_sample_data()

    model = build_logistic_regression()

    trained = train_model(model, X_train, y_train)

    preds = predict(trained, X_test)

    assert len(preds) == len(y_test)


def test_predict_proba():

    X_train, X_test, y_train, y_test = create_sample_data()

    model = build_logistic_regression()

    trained = train_model(model, X_train, y_train)

    probs = predict_proba(trained, X_test)

    assert probs.shape[0] == len(y_test)


def test_fit_and_predict():

    X_train, X_test, y_train, y_test = create_sample_data()

    model = build_logistic_regression()

    results = fit_and_predict(model, X_train, y_train, X_test)

    assert "model" in results
    assert "predictions" in results
    assert "probabilities" in results
