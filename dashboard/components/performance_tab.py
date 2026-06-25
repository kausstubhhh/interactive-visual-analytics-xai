from dash import dcc, html
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly"

EXPORT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "exports"
    / "evaluation_summary.csv"
)

print("USING PERFORMANCE_TAB")

def load_evaluation_data() -> pd.DataFrame:

    if not EXPORT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EXPORT_FILE}"
        )

    return pd.read_csv(EXPORT_FILE)


def create_metric_figure(
    df,
    metric,
    title
):
    print("Current template:", pio.templates.default)
    print("=" * 60)
    print(title)
    print(df)
    print(df.dtypes)
    print(type(df))
    print("=" * 60)

    fig = go.Figure()

    fig.add_bar(
        x=df["model"],
        y=df[metric]
    )

    fig.update_layout(
        title=title,
        height=450
    )

    return fig


def create_performance_tab():

    df = load_evaluation_data()

    return html.Div(
        [
            html.H2(
                "Task 1: Performance Comparison"
            ),

            html.H3(
                "Question"
            ),

            html.P(
                "Which model performs best on the uploaded dataset?"
            ),

            dcc.Graph(
                id="accuracy-chart"
            ),

            dcc.Graph(
                id="precision-chart"
            ),

            dcc.Graph(
                id="recall-chart"
            ),

            dcc.Graph(
                id="f1-chart"
            ),

            dcc.Graph(
                id="roc-chart"
            )
        ]
    )