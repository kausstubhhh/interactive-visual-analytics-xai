from dash import dcc, html
from dash import dash_table
from ui_components import create_card, create_analysis_card, create_metric_card


def create_feature_importance_tab():

    return html.Div(
        [
            html.H2("Feature Importance Analysis"),
            html.P(
                (
                    "Explore which features contributed most to the "
                    "predictions made by each classification model "
                    "using global SHAP feature importance."
                )
            ),
            create_analysis_card(
                title="Feature Importance Summary",
                component_id="feature-summary",
                icon="📊",
            ),
            html.Div(id="feature-metric-cards"),
            create_card(
                "Analysis Controls",
                html.Div(
                    [
                        html.Label("Model"),
                        dcc.Dropdown(
                            id="model-dropdown",
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
                            id="topn-dropdown",
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
            create_card(
                "Global SHAP Feature Importance", dcc.Graph(id="shap-importance-chart")
            ),
            create_card(
                "Feature Importance Values",
                dash_table.DataTable(
                    id="shap-table",
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "10px"},
                    style_header={"fontWeight": "bold"},
                ),
            ),
        ]
    )


def build_feature_metric_cards(df, model, top_n):
    top_feature = df.iloc[0]
    return html.Div(
        [
            create_metric_card("Top Feature", top_feature["feature"]),
            create_metric_card("Importance", f"{top_feature['importance']:.4f}"),
            create_metric_card("Model", model.replace("_", " ").title()),
            create_metric_card("Features Shown", str(top_n)),
        ],
        className="performance-summary-row",
    )


def generate_feature_summary(df, model):

    top_feature = df.iloc[0]

    return (
        f"{top_feature['feature']} is the most influential feature "
        f"for the {model.replace('_', ' ').title()} model "
        f"based on the mean absolute SHAP value "
        f"({top_feature['importance']:.4f})."
    )
