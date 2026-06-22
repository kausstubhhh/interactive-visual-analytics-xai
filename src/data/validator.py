import pandas as pd


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    High-level dataset information.
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


def get_column_names(df: pd.DataFrame) -> list:
    """
    Return all column names.
    """

    return list(df.columns)


def column_types(df: pd.DataFrame) -> pd.Series:
    """
    Return datatype information.
    """

    return df.dtypes


def missing_values_report(df: pd.DataFrame) -> pd.Series:
    """
    Missing values per column.
    """

    return df.isnull().sum()


def duplicate_count(df: pd.DataFrame) -> int:
    """
    Count duplicate rows.
    """

    return int(df.duplicated().sum())