from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx"
}


def load_dataset(file_path):
    """
    Generic dataset loader.
    Supports CSV, XLS and XLSX files.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    if suffix == ".csv":
        return pd.read_csv(file_path)

    return pd.read_excel(file_path)


def load_demo_dataset(dataset_name):
    """
    Load built-in evaluation datasets.
    """

    datasets = {
        "heloc": RAW_DATA_DIR / "heloc_dataset_v1.csv",
        "bank": RAW_DATA_DIR / "bank-additional-full.xlsx"
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    return load_dataset(
        datasets[dataset_name]
    )