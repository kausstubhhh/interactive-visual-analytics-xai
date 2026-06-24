from dash import dcc, html
import pandas as pd
import plotly.express as px
from pathlib import Path


EXPORT_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "exports"
    / "evaluation_summary.csv"
)


def load_evaluation_data() -> pd.DataFrame:

    if not EXPORT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EXPORT_FILE}"
        )

    return pd.read_csv(EXPORT_FILE)


def create_metric_figure(
    df: pd.DataFrame,
    metric: str,
    title: str
):

    fig = px.bar(
        df,
        x="dataset",
        y=metric,
        color="model",
        barmode="group",
        text_auto=".3f",
        title=title
    )

    fig.update_layout(
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

            dcc.Graph(
                figure=create_metric_figure(
                    df,
                    "accuracy",
                    "Accuracy Comparison"
                )
            ),

            dcc.Graph(
                figure=create_metric_figure(
                    df,
                    "precision",
                    "Precision Comparison"
                )
            ),

            dcc.Graph(
                figure=create_metric_figure(
                    df,
                    "recall",
                    "Recall Comparison"
                )
            ),

            dcc.Graph(
                figure=create_metric_figure(
                    df,
                    "f1_score",
                    "F1 Score Comparison"
                )
            ),

            dcc.Graph(
                figure=create_metric_figure(
                    df,
                    "roc_auc",
                    "ROC-AUC Comparison"
                )
            )
        ]
    )