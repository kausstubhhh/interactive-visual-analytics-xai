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

def get_dataset_profile(
    df: pd.DataFrame
) -> dict:
    """
    Generate detailed dataset profile.
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
            if pd.api.types.is_numeric_dtype(df[column])
            else "Categorical"
        )
        sample_values = list(
            unique_values[:5]
        )
        recommendation = ""
        # Binary classification target
        if unique_count == 2:
            recommendation = "⭐ Recommended"
            recommended_targets.append(
                column
            )

        # Possible future multiclass support
        elif 3 <= unique_count <= 10:
            recommendation = "⚠ Future Extension"
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