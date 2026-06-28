import plotly.graph_objects as go

from theme import (
    PRIMARY,
    ERROR
)


def confusion_matrix_figure(metrics):

    fig = go.Figure(
        data=go.Heatmap(
            z=[
                [
                    metrics["true_negatives"],
                    metrics["false_positives"]
                ],
                [
                    metrics["false_negatives"],
                    metrics["true_positives"]
                ]
            ],
            x=[
                "Predicted Negative",
                "Predicted Positive"
            ],
            y=[
                "Actual Negative",
                "Actual Positive"
            ],
            text=[
                [
                    metrics["true_negatives"],
                    metrics["false_positives"]
                ],
                [
                    metrics["false_negatives"],
                    metrics["true_positives"]
                ]
            ],
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    fig.update_layout(
        title=None,
        height=450
    )

    return fig


def error_breakdown_figure(metrics):

    fig = go.Figure()

    fig.add_bar(
        x=[
            "False Positives",
            "False Negatives"
        ],
        y=[
            metrics["false_positives"],
            metrics["false_negatives"]
        ],
        marker_color=[
            ERROR,
            PRIMARY
        ]
    )

    fig.update_layout(
        title=None,
        height=420
    )

    return fig