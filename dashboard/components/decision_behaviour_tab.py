from dash import dcc, html
from dash import dash_table

from ui_components import create_card, create_analysis_card, create_metric_card


def create_decision_behaviour_tab():

    return html.Div(
        [
            html.H2("Decision Behaviour Analysis"),
            html.P(
                (
                    "Investigate how individual features contribute "
                    "to a single prediction using local SHAP explanations."
                )
            ),
            create_analysis_card(
                title="Decision Behaviour Summary",
                component_id="decision-summary",
                icon="🧠",
            ),
            html.Div(id="decision-metric-cards"),
            create_card(
                "Analysis Controls",
                html.Div(
                    [
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="decision-model-dropdown",
                            options=[
                                {
                                    "label": "Logistic Regression",
                                    "value": "logistic_regression",
                                },
                                {"label": "Random Forest", "value": "random_forest"},
                            ],
                            value="logistic_regression",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("Top Features"),
                        dcc.Dropdown(
                            id="decision-topn-dropdown",
                            options=[
                                {"label": "Top 5", "value": 5},
                                {"label": "Top 10", "value": 10},
                                {"label": "Top 15", "value": 15},
                                {"label": "Top 20", "value": 20},
                            ],
                            value=10,
                            clearable=False,
                        ),
                    ]
                ),
            ),
            create_card("Local SHAP Explanation", dcc.Graph(id="decision-chart")),
            create_card(
                "Feature Contributions",
                dash_table.DataTable(
                    id="decision-table",
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    style_header={"fontWeight": "bold"},
                ),
            ),
        ]
    )


def build_decision_metric_cards(df, model, top_n):
    strongest = df.iloc[0]
    return html.Div(
        [
            create_metric_card("Strongest Feature", strongest["feature"]),
            create_metric_card("SHAP Value", f"{strongest['SHAP Value']:.3f}"),
            create_metric_card("Model", model.replace("_", " ").title()),
            create_metric_card("Features Shown", str(top_n)),
        ],
        className="performance-summary-row",
    )


def generate_decision_summary(df, model):
    strongest = df.iloc[0]
    return (
        f"{strongest['feature']} had the greatest influence "
        f"on the selected prediction for the "
        f"{model.replace('_', ' ').title()} model "
        f"with a SHAP contribution of "
        f"{strongest['SHAP Value']:.3f}."
    )
