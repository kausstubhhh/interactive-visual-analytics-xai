from src.evaluation.evaluator import evaluate_model


def test_evaluator_returns_expected_sections():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]

    results = evaluate_model(y_true, y_pred, y_proba)

    assert "metrics" in results
    assert "confusion_matrix" in results
    assert "confusion_values" in results
    assert "classification_report" in results


def test_evaluator_metrics_accuracy():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]

    results = evaluate_model(y_true, y_pred, y_proba)

    assert results["metrics"]["accuracy"] == 1.0
