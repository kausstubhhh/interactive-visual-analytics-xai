from pathlib import Path


def export_feature_importance(
    importance_df,
    output_path
):
    """
    Export SHAP feature importance
    table to CSV.

    Parameters
    ----------
    importance_df : pandas.DataFrame

    output_path : str | Path
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        output_path,
        index=False
    )

    return output_path