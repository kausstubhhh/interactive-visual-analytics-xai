from pathlib import Path

from src.data.loader import load_dataset
from src.data.schema_detector import detect_schema
from src.data.preprocess import prepare_dataset

from src.models.logistic_regression import build_logistic_regression

from src.models.random_forest import build_random_forest

from src.models.trainer import fit_and_predict

DATA_DIR = Path("data/raw")


DATASETS = {
    "HELOC": {"path": DATA_DIR / "heloc_dataset_v1.csv", "target": "RiskPerformance"},
    "BANK": {"path": DATA_DIR / "bank-additional-full.xlsx", "target": "y"},
}


def train_dataset(dataset_name, dataset_path, target_column):

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    df = load_dataset(dataset_path)

    schema = detect_schema(df, target_column)

    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
        df, target_column, schema
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
    }

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        results = fit_and_predict(model, X_train_processed, y_train, X_test_processed)

        print("Done")

        print(f"Predictions: " f"{results['predictions'].shape}")

        print(f"Probabilities: " f"{results['probabilities'].shape}")


if __name__ == "__main__":

    for dataset_name, config in DATASETS.items():

        train_dataset(dataset_name, config["path"], config["target"])
