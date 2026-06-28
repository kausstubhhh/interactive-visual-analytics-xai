"""
XAI service layer.

Coordinates SHAP explainability
analysis and export generation.
"""

from pathlib import Path

from src.explainability.shap_explainer import build_shap_explainer

from src.explainability.importance import calculate_feature_importance

from src.explainability.local_explanations import get_local_explanation

from src.explainability.exporter import export_feature_importance

SHAP_EXPORT_DIR = Path("data") / "exports" / "shap"

LOCAL_EXPORT_DIR = Path("data") / "exports" / "local_explanations"


def generate_shap_values(model, X_background, X_explain):
    """
    Generate SHAP values.
    """

    explainer = build_shap_explainer(model=model, background_data=X_background)

    shap_values = explainer(X_explain, check_additivity=False)

    return shap_values


def generate_feature_importance(shap_values, feature_names):
    """
    Generate global importance table.
    """

    return calculate_feature_importance(shap_values, feature_names)


def generate_local_explanation(shap_values, feature_names, instance_index=0):
    """
    Generate local explanation.
    """

    return get_local_explanation(
        shap_values=shap_values,
        feature_names=feature_names,
        instance_index=instance_index,
    )


def export_importance_table(importance_df, dataset_name, model_name):
    """
    Export SHAP importance.
    """

    output_file = SHAP_EXPORT_DIR / f"{dataset_name}_{model_name}_importance.csv"

    return export_feature_importance(
        importance_df=importance_df, output_path=output_file
    )


def export_local_explanation(
    explanation_df, dataset_name, model_name, instance_index=0
):
    """
    Export local explanation.
    """

    output_file = (
        LOCAL_EXPORT_DIR / f"{dataset_name}_{model_name}_instance_{instance_index}.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    explanation_df.to_csv(output_file, index=False)

    return output_file


def run_xai_analysis(dataset_name, model_name, model, X_train, X_test, feature_names):
    """
    Complete XAI workflow.
    """

    MAX_EXPLANATION_SAMPLES = 100

    X_explain = X_test[:MAX_EXPLANATION_SAMPLES]

    print(f"\nRunning XAI for {model_name}")

    print(f"Using {len(X_explain)} samples " f"for SHAP analysis")

    shap_values = generate_shap_values(
        model=model, X_background=X_train, X_explain=X_explain
    )

    print("SHAP values generated")

    importance_df = generate_feature_importance(shap_values, feature_names)

    print("Feature importance generated")

    local_df = generate_local_explanation(shap_values, feature_names, instance_index=0)

    print("Local explanation generated")

    export_importance_table(importance_df, dataset_name, model_name)

    export_local_explanation(local_df, dataset_name, model_name, instance_index=0)

    print(f"XAI completed for {model_name}")

    return {
        "shap_values": shap_values,
        "importance": importance_df,
        "local_explanation": local_df,
    }
