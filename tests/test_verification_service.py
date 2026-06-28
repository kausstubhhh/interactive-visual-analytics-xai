import pandas as pd

from src.services.verification_service import (
    verify_dataset,
    analyse_columns,
    validate_candidate_targets,
)


def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
            "approved": [1, 0, 1, 0],
        }
    )


def test_verify_dataset():
    df = sample_dataframe()

    report = verify_dataset(df)

    assert report["ready"] is True
    assert "checks" in report
    assert "columns" in report
    assert "compatible_targets" in report


def test_analyse_columns():
    df = sample_dataframe()

    columns = analyse_columns(df)

    assert len(columns) == 3

    assert columns[0]["name"] == "age"


def test_validate_candidate_targets():
    df = sample_dataframe()

    profiles = analyse_columns(df)

    candidates = validate_candidate_targets(
        df,
        profiles,
    )

    assert isinstance(candidates, list)

    assert any(
        c["column"] == "approved"
        for c in candidates
    )