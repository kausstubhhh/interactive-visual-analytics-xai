import plotly.graph_objects as go

from theme import (
    SUCCESS,
    ERROR
)


def local_shap_chart(df):

    colours = [
        SUCCESS if value >= 0 else ERROR
        for value in df["SHAP Value"]
    ]

    fig = go.Figure()

    fig.add_bar(
        x=df["SHAP Value"],
        y=df["feature"],
        orientation="h",
        marker_color=colours,
        text=df["SHAP Value"].round(3),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP Value: %{x:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title=None,
        height=max(500, len(df) * 35),
        margin=dict(
            t=20,
            l=20,
            r=20,
            b=20
        ),
        xaxis_title="SHAP Contribution",
        yaxis_title="Feature",
        yaxis=dict(
            autorange="reversed"
        )
    )

    return fig