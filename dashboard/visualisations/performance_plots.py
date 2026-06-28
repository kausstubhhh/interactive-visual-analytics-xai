import plotly.graph_objects as go

from theme import PRIMARY, SUCCESS, BACKGROUND, TEXT_PRIMARY


def empty_figure(message):

    fig = go.Figure()

    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(plot_bgcolor=BACKGROUND, paper_bgcolor=BACKGROUND, height=450)
    return fig


def performance_bar_chart(df, metric, title):
    colours = [PRIMARY, SUCCESS]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[m.replace("_", " ").title() for m in df["model"]],
            y=df[metric],
            marker_color=colours,
            text=[f"{v:.3f}" for v in df[metric]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        height=450,
        template="plotly_white",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor="white",
        font=dict(color=TEXT_PRIMARY),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(range=[0, 1])
    return fig
