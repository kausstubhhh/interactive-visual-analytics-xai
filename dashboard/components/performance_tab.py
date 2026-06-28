from dash import dcc, html
import pandas as pd
from pathlib import Path
import plotly.io as pio

from ui_components import create_analysis_card, create_card, create_metric_card

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
        raise FileNotFoundError(f"Evaluation file not found: {EXPORT_FILE}")

    return pd.read_csv(EXPORT_FILE)


def build_metric_cards(best_model):

    return html.Div(
        [
            create_metric_card(
                "Best Model", best_model["model"].replace("_", " ").title()
            ),
            create_metric_card("Accuracy", f"{best_model['accuracy']:.3f}"),
            create_metric_card("F1 Score", f"{best_model['f1_score']:.3f}"),
            create_metric_card("ROC-AUC", f"{best_model['roc_auc']:.3f}"),
        ],
        className="performance-summary-row",
    )


def create_performance_tab():

    return html.Div(
        [
            html.H2("Performance Comparison"),
            html.P(
                (
                    "Compare the predictive performance of the trained "
                    "classification models using standard evaluation metrics."
                )
            ),
            create_analysis_card(
                title="Performance Summary",
                component_id="performance-summary",
                icon="🏆",
            ),
            html.Div(id="performance-metric-cards"),
            create_card(
                "Performance Explorer",
                html.Div(
                    [
                        html.Label("Performance Metric"),
                        dcc.Dropdown(
                            id="metric-selector",
                            options=[
                                {"label": "Accuracy", "value": "accuracy"},
                                {"label": "Precision", "value": "precision"},
                                {"label": "Recall", "value": "recall"},
                                {"label": "F1 Score", "value": "f1_score"},
                                {"label": "ROC-AUC", "value": "roc_auc"},
                            ],
                            value="accuracy",
                            clearable=False,
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                dcc.Graph(id="performance-comparison-chart"),
            ),
            create_card(
                "Model Comparison", html.Div(id="performance-comparison-table")
            ),
            create_card("ROC Curve", dcc.Graph(id="roc-curve-chart")),
        ]
    )
