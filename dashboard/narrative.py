def generate_performance_summary(df):
    """
    Generate a narrative summary of model performance.
    """

    best_model = (
        df.sort_values(
            by="f1_score",
            ascending=False
        )
        .iloc[0]
    )

    other_model = (
        df.sort_values(
            by="f1_score",
            ascending=False
        )
        .iloc[1]
    )

    difference = (
        best_model["f1_score"]
        - other_model["f1_score"]
    )

    if difference < 0.02:

        comparison = (
            "The improvement over "
            f"{other_model['model'].replace('_', ' ').title()} "
            "is relatively small, indicating that both models "
            "perform similarly on this dataset."
        )

    else:

        comparison = (
            f"{best_model['model'].replace('_', ' ').title()} "
            "provides a noticeable improvement over the "
            "alternative model."
        )

    summary = (

        f"{best_model['model'].replace('_', ' ').title()} "

        "achieved the strongest overall predictive "

        f"performance based on the F1-score "

        f"({best_model['f1_score']:.3f}) "

        f"and ROC-AUC "

        f"({best_model['roc_auc']:.3f}). "

        + comparison

    )

    return summary