from src.evaluation.metrics import calculate_metrics


def test_calculate_metrics_returns_all_metrics():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]

    results = calculate_metrics(y_true, y_pred, y_proba)

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "roc_auc" in results


def test_metrics_are_perfect_for_perfect_predictions():

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]

    results = calculate_metrics(y_true, y_pred, y_proba)

    assert results["accuracy"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["roc_auc"] == 1.0
