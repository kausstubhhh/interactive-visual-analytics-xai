import pandas as pd


def detect_schema(
    df: pd.DataFrame,
    target_column: str
) -> dict:
    """
    Detect dataset structure.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    feature_df = df.drop(columns=[target_column])

    categorical_columns = list(
        feature_df.select_dtypes(
            include=["object"]
        ).columns
    )

    numerical_columns = list(
        feature_df.select_dtypes(
            exclude=["object"]
        ).columns
    )

    return {
        "target": target_column,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns
    }