import plotly.express as px

from theme import PRIMARY


def feature_importance_chart(df):

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        text="importance"
    )

    fig.update_traces(
        marker_color=PRIMARY,
        texttemplate="%{text:.3f}",
        textposition="outside"
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
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    return fig