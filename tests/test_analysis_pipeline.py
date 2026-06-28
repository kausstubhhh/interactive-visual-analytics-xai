import pandas as pd

from src.services.analysis_pipeline import (
    run_analysis_pipeline,
)


def test_analysis_pipeline_returns_expected_sections():
    df = pd.read_csv("data/raw/heloc_dataset_v1.csv")

    result = run_analysis_pipeline(
        df=df,
        dataset_name="pytest",
        target_column="RiskPerformance",
    )

    assert isinstance(result, dict)

    assert "schema" in result
    assert "evaluation" in result
    assert "xai" in result


def test_pipeline_returns_dictionary():
    df = pd.read_csv("data/raw/heloc_dataset_v1.csv")

    result = run_analysis_pipeline(
        df=df,
        dataset_name="pytest",
        target_column="RiskPerformance",
    )

    assert isinstance(result, dict)