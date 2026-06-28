from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SHAP_EXPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "shap"
)

ERROR_EXPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "errors"
)

LOCAL_EXPLANATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "exports"
    / "local_explanations"
)

# ============================================================
# SHAP FUNCTIONS
# ============================================================

def load_shap_file(
    dataset: str,
    model: str
) -> pd.DataFrame:
    """
    Load SHAP importance CSV.
    """

    filename = (
        f"{dataset}_{model}_importance.csv"
    )

    file_path = (
        SHAP_EXPORT_DIR / filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"SHAP file not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# ERROR FUNCTIONS
# ============================================================

def load_error_file(
    dataset: str,
    model: str
) -> pd.DataFrame:
    """
    Load misclassification summary CSV.
    """

    filename = (
        f"{dataset}_{model}_errors.csv"
    )

    file_path = (
        ERROR_EXPORT_DIR / filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Error file not found: {file_path}"
        )

    return pd.read_csv(file_path)

def load_confusion_file(
    dataset,
    model
):
    """
    Load confusion matrix CSV.
    """

    filename = (
        f"{dataset}_{model}_confusion.csv"
    )

    CONFUSION_EXPORT_DIR = (
        PROJECT_ROOT
        / "data"
        / "exports"
        / "confusion"
    )

    file_path = CONFUSION_EXPORT_DIR / filename

    if not file_path.exists():

        raise FileNotFoundError(
            f"Confusion file not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        index_col=0
    )


# ============================================================
# CALLBACKS
# ============================================================
def load_local_explanation_file(
    dataset: str,
    model: str
) -> pd.DataFrame:
    """
    Load local explanation CSV.
    """

    filename = (
        f"{dataset}_{model}_instance_0.csv"
    )

    file_path = (
        LOCAL_EXPLANATION_DIR
        / filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Local explanation file not found: {file_path}"
        )

    return pd.read_csv(file_path)