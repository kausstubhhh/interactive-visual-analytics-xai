from pathlib import Path

import pandas as pd

from src.data.schema_detector import (
    detect_schema
)


def load_uploaded_dataset(
    file_path: str | Path
) -> pd.DataFrame:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    if file_path.suffix.lower() in [
        ".xlsx",
        ".xls"
    ]:
        return pd.read_excel(file_path)

    raise ValueError(
        f"Unsupported file type: {file_path.suffix}"
    )


def preview_dataset(
    df: pd.DataFrame,
    rows: int = 10
) -> pd.DataFrame:
    """
    Return first rows for preview.
    """

    return df.head(rows)


def get_dataset_summary(
    df: pd.DataFrame
) -> dict:
    """
    Generate dataset summary.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "column_names": list(
            df.columns
        )
    }


def detect_dataset_schema(
    df: pd.DataFrame,
    target_column: str
) -> dict:
    """
    Wrapper around schema detector.
    """

    return detect_schema(
        df=df,
        target_column=target_column
    )