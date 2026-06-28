from pathlib import Path

from src.data.loader import load_dataset
from src.data.schema_detector import detect_schema
from src.data.preprocess import prepare_dataset

from src.models.logistic_regression import build_logistic_regression

from src.models.random_forest import build_random_forest

from src.models.trainer import fit_and_predict

from src.explainability.shap_explainer import build_shap_explainer

from src.explainability.shap_generator import generate_shap_values

from src.explainability.local_explanations import get_local_explanation

DATA_DIR = Path("data/raw")

EXPORT_DIR = Path("data/exports/local_explanations")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "HELOC": {"path": DATA_DIR / "heloc_dataset_v1.csv", "target": "RiskPerformance"},
    "BANK": {"path": DATA_DIR / "bank-additional-full.xlsx", "target": "y"},
}


MODELS = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
}


INSTANCE_INDEX = 0


def generate_explanations(dataset_name, dataset_path, target_column):

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

    feature_names = preprocessor.get_feature_names_out()

    for model_name, model_builder in MODELS.items():

        print(f"\nGenerating local explanation " f"for {model_name}")

        model = model_builder()

        fit_and_predict(model, X_train_processed, y_train, X_test_processed)

        explainer = build_shap_explainer(model, X_train_processed)

        shap_values = generate_shap_values(
            explainer, X_test_processed[INSTANCE_INDEX : INSTANCE_INDEX + 1]
        )

        explanation_df = get_local_explanation(shap_values, feature_names, 0)

        output_file = (
            EXPORT_DIR / f"{dataset_name.lower()}_" f"{model_name}_" f"instance_0.csv"
        )

        explanation_df.to_csv(output_file, index=False)

        print(f"Exported: {output_file}")

        print(explanation_df.head(10).to_string(index=False))


if __name__ == "__main__":

    for dataset_name, config in DATASETS.items():

        generate_explanations(dataset_name, config["path"], config["target"])
