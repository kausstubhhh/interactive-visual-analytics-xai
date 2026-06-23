import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest


def test_logistic_regression_creation():
    model = build_logistic_regression()

    assert model is not None


def test_random_forest_creation():
    model = build_random_forest()

    assert model is not None