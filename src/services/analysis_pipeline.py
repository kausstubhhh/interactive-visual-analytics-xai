"""
End-to-end analysis pipeline.

Dataset
    ↓
Schema Detection
    ↓
Preprocessing
    ↓
Model Training
    ↓
Evaluation
    ↓
Explainability
"""

from src.services.dataset_service import detect_dataset_schema

from src.services.model_service import train_models

from src.services.evaluation_service import evaluate_all_models

from src.services.xai_service import run_xai_analysis

from src.data.preprocess import prepare_dataset


def run_analysis_pipeline(df, dataset_name, target_column):
    """
    Complete analysis workflow.
    """

    # ---------------------------------
    # Schema Detection
    # ---------------------------------

    schema = detect_dataset_schema(df=df, target_column=target_column)

    # ---------------------------------
    # Preprocessing
    # ---------------------------------

    X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(
        df=df, target_column=target_column, schema=schema
    )

    # ---------------------------------
    # Transform Features
    # ---------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # ---------------------------------
    # Train Models
    # ---------------------------------

    model_results = train_models(
        X_train=X_train_processed, y_train=y_train, X_test=X_test_processed
    )

    # ---------------------------------
    # Evaluation
    # ---------------------------------

    evaluation_results = evaluate_all_models(
        dataset_name=dataset_name, y_true=y_test, model_results=model_results
    )

    # ---------------------------------
    # Explainability
    # ---------------------------------

    xai_results = {}

    for model_name, result in model_results.items():

        model = result["model"]

        xai_results[model_name] = run_xai_analysis(
            dataset_name=dataset_name,
            model_name=model_name,
            model=model,
            X_train=X_train_processed,
            X_test=X_test_processed,
            feature_names=feature_names,
        )

    return {"schema": schema, "evaluation": evaluation_results, "xai": xai_results}
