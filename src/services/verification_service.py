"""
Dataset verification service.

Validates uploaded datasets before
running the analysis pipeline.
"""
import pandas as pd

def verify_dataset(
    df: pd.DataFrame
) -> dict:
    """
    Verify whether the uploaded dataset
    is suitable for analysis.
    """
    report = {
        "ready": True,
        "checks": [],
        "warnings": []

    }

    # Basic dataset checks

    report["checks"].append(
        {
            "name": "Dataset Loaded",
            "passed": True,
            "message": "Dataset successfully loaded."
        }
    )
    report["checks"].append(
        {
            "name": "Rows",
            "passed": len(df) >= 50,
            "message": f"{len(df)} rows detected."
        }
    )
    report["checks"].append(
        {
            "name": "Columns",
            "passed": len(df.columns) >= 2,
            "message": f"{len(df.columns)} columns detected."
        }
    )
    missing = int(
        df.isna().sum().sum()
    )
    report["checks"].append(
        {
            "name": "Missing Values",
            "passed": True,
            "message": f"{missing} missing values detected."
        }
    )
    report["columns"] = analyse_columns(
        df
    )
    
    
    report["compatible_targets"] = [
        candidate["column"]
        for candidate in validate_candidate_targets(
            df,
            report["columns"]
        )
        if candidate["valid"]
    ]

    report["target_validation"] = (
        validate_candidate_targets(
            df,
            report["columns"]
        )
    )
    return report

def analyse_columns(
    df: pd.DataFrame
) -> list:
    """
    Analyse every column in the dataset.
    """
    columns = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        unique_count = (
            series.dropna()
            .nunique()
        )
        missing_count = int(
            series.isna().sum()
        )
        missing_percent = (
            missing_count
            / total_rows
        ) * 100
        is_numeric = (
            pd.api.types.is_numeric_dtype(
                series
            )
        )
        is_constant = (
            unique_count <= 1
        )
        unique_ratio = (
            unique_count
            / total_rows
        )
        looks_like_identifier = (
            unique_ratio > 0.95
        )

        columns.append(
            {
                "name": column,
                "dtype":
                    "Numerical"
                    if is_numeric
                    else "Categorical",
                "unique_values":
                    unique_count,
                "missing_values":
                    missing_count,
                "missing_percent":
                    round(
                        missing_percent,
                        2
                    ),
                "constant":
                    is_constant,
                "identifier":
                    looks_like_identifier
            }
        )
    return columns

def validate_candidate_targets(
    df: pd.DataFrame,
    column_profiles: list
) -> list:
    """
    Validate candidate target columns.
    """
    candidates = []

    for profile in column_profiles:
        column = profile["name"]
        series = df[column]
        reasons = []
        valid = True

        if profile["identifier"]:
            valid = False
            reasons.append(
                "Identifier column"
            )

        if profile["constant"]:
            valid = False
            reasons.append(
                "Constant column"
            )

        if profile["unique_values"] != 2:
            valid = False
            reasons.append(
                "Not binary"
            )

        if valid:
            class_counts = (
                series
                .value_counts(
                    dropna=False
                )
            )

            if (
                class_counts.min()
                < 2
            ):
                valid = False
                reasons.append(
                    "Insufficient samples per class"
                )
        candidates.append(
            {
                "column": column,
                "valid": valid,
                "reasons": reasons
            }
        )
    return candidates