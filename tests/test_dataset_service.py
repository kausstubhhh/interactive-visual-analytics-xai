import pandas as pd
import pytest

from src.services.dataset_service import (
    load_uploaded_dataset,
    preview_dataset,
    get_dataset_summary,
    detect_dataset_schema,
    get_dataset_profile,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
            "approved": [1, 0, 1, 0],
            "city": ["A", "B", "A", "C"],
        }
    )


# ==========================================================
# load_uploaded_dataset
# ==========================================================

def test_load_uploaded_dataset_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    loaded = load_uploaded_dataset(file_path)

    pd.testing.assert_frame_equal(df, loaded)


def test_load_uploaded_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_uploaded_dataset("does_not_exist.csv")


def test_load_uploaded_dataset_invalid_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("dummy")

    with pytest.raises(ValueError):
        load_uploaded_dataset(file_path)


# ==========================================================
# preview_dataset
# ==========================================================

def test_preview_dataset(sample_dataframe):
    preview = preview_dataset(sample_dataframe, rows=2)

    assert len(preview) == 2
    assert list(preview.columns) == list(sample_dataframe.columns)


# ==========================================================
# get_dataset_summary
# ==========================================================

def test_get_dataset_summary(sample_dataframe):
    summary = get_dataset_summary(sample_dataframe)

    assert summary["rows"] == 4
    assert summary["columns"] == 4
    assert summary["missing_values"] == 0
    assert "approved" in summary["column_names"]


# ==========================================================
# detect_dataset_schema
# ==========================================================

def test_detect_dataset_schema(sample_dataframe):
    schema = detect_dataset_schema(
        sample_dataframe,
        target_column="approved",
    )

    assert isinstance(schema, dict)


# ==========================================================
# get_dataset_profile
# ==========================================================

def test_get_dataset_profile(sample_dataframe):
    profile = get_dataset_profile(sample_dataframe)

    assert profile["rows"] == 4
    assert profile["columns"] == 4
    assert profile["missing_values"] == 0

    assert "recommended_targets" in profile
    assert "approved" in profile["recommended_targets"]

    assert len(profile["column_profiles"]) == 4


def test_dataset_profile_contains_column_information(sample_dataframe):
    profile = get_dataset_profile(sample_dataframe)

    first_column = profile["column_profiles"][0]

    assert "column" in first_column
    assert "type" in first_column
    assert "unique_count" in first_column
    assert "sample_values" in first_column
    assert "recommendation" in first_column