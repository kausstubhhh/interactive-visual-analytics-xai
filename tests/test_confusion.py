from src.evaluation.confusion import (
    calculate_confusion_matrix,
    extract_confusion_values
)


def test_confusion_matrix_shape():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred
    )

    assert matrix.shape == (2, 2)


def test_confusion_values():

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    values = extract_confusion_values(
        y_true,
        y_pred
    )

    assert values["tn"] == 1
    assert values["fp"] == 1
    assert values["fn"] == 1
    assert values["tp"] == 1