from pathlib import Path
import pandas as pd
from src.data.schema_detector import (
    detect_schema
)

def load_uploaded_dataset(
    file_path: str | Path
) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset.
    """

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

def get_dataset_profile(
    df: pd.DataFrame
) -> dict:
    """
    Generate a detailed dataset profile for the
    Dataset Management page.
    """

    column_profiles = []
    recommended_targets = []

    for column in df.columns:
        unique_values = (
            df[column]
            .dropna()
            .unique()
        )
        unique_count = len(
            unique_values
        )
        dtype = (
            "Numerical"
            if pd.api.types.is_numeric_dtype(
                df[column]
            )
            else "Categorical"
        )
        sample_values = list(
            unique_values[:5]
        )
        recommendation = ""

        # Binary classification target
        if unique_count == 2:
            recommendation = "Compatible"
            recommended_targets.append(column)
        else:
            recommendation = "Feature"
 
        column_profiles.append(
            {
                "column": column,
                "type": dtype,
                "unique_count": unique_count,
                "sample_values": sample_values,
                "recommendation": recommendation
            }
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "numerical_columns": len(
            df.select_dtypes(
                include="number"
            ).columns
        ),
        "categorical_columns": len(
            df.select_dtypes(
                exclude="number"
            ).columns
        ),
        "recommended_targets": recommended_targets,
        "column_profiles": column_profiles
    }