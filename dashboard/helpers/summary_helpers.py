def generate_error_summary(metrics):

    return (
        f"The selected model produced "
        f"{int(metrics['total_errors'])} incorrect predictions "
        f"({metrics['error_rate']:.1%} error rate). "
        f"There were "
        f"{int(metrics['false_positives'])} false positives and "
        f"{int(metrics['false_negatives'])} false negatives."
    )
