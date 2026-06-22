import pandas as pd


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return high-level dataset information.
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


def missing_values_report(df: pd.DataFrame):
    """
    Missing values per column.
    """

    return df.isnull().sum()


def duplicate_count(df: pd.DataFrame):
    """
    Number of duplicate rows.
    """

    return int(df.duplicated().sum())


def column_types(df: pd.DataFrame):
    """
    Column datatype information.
    """

    return df.dtypes