import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def encode_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Encode any binary target to 0/1.
    """

    df = df.copy()

    if df[target_column].nunique() == 2:

        encoder = LabelEncoder()

        df[target_column] = encoder.fit_transform(df[target_column])

    return df


def clean_heloc_special_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    HELOC uses -7, -8 and -9
    as special missing-value codes.
    """

    df = df.copy()

    for value in [-7, -8, -9]:
        df = df.replace(value, None)

    return df


def build_preprocessor(numerical_columns, categorical_columns):
    """
    Build sklearn preprocessing pipeline.
    """

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numerical", numeric_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def prepare_dataset(df, target_column, schema, test_size=0.2, random_state=42):
    """
    Full preprocessing workflow.
    """

    df = encode_target(df, target_column)

    # Validate Target

    if df[target_column].nunique() != 2:
        raise ValueError(
            "This dashboard currently supports binary classification datasets only."
        )

    if target_column == "RiskPerformance":
        df = clean_heloc_special_values(df)

    X = df.drop(columns=[target_column])

    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(
        schema["numerical_columns"], schema["categorical_columns"]
    )

    return (X_train, X_test, y_train, y_test, preprocessor)
