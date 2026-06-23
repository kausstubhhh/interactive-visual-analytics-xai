from src.evaluation.misclassification import (
    get_false_positives,
    get_false_negatives,
    calculate_error_summary
)


def test_false_positives():

    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 1, 0]

    result = get_false_positives(
        y_true,
        y_pred
    )

    assert len(result) == 1


def test_false_negatives():

    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 1, 0]

    result = get_false_negatives(
        y_true,
        y_pred
    )

    assert len(result) == 1


def test_error_summary():

    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 1, 0]

    result = calculate_error_summary(
        y_true,
        y_pred
    )

    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["total_errors"] == 2