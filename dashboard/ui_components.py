from dash import html


def create_card(title, *children, class_name=""):
    classes = "dashboard-card"
    if class_name:
        classes += f" {class_name}"

    return html.Div(
        [html.H3(title, className="card-title"), *children], className=classes
    )


def create_analysis_card(title: str, component_id: str, icon: str = "📊"):
    """
    Create a reusable analysis summary card.

    Parameters
    ----------
    title
        Card title.

    component_id
        Dash component id that callbacks will update.

    icon
        Emoji shown beside the title.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon, className="analysis-icon"),
                    html.H3(title, className="analysis-title"),
                ],
                className="analysis-header",
            ),
            html.Div(
                "Analysis results will appear here after running the model.",
                id=component_id,
                className="analysis-content",
            ),
        ],
        className="analysis-card",
    )


def create_metric_card(title: str, value: str, colour: str = "#4E79A7"):

    return html.Div(
        [
            html.Div(title.upper(), className="metric-card-title"),
            html.Div(value, className="metric-card-value", style={"color": colour}),
        ],
        className="performance-metric-card",
    )
