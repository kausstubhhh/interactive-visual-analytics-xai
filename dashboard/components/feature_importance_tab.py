from dash import dcc, html
from dash import dash_table


def create_feature_importance_tab():

    return html.Div(
        [

            html.H2(
                "Task 2: Feature Importance Analysis"
            ),

            html.Br(),

            html.Label("Model"),

            dcc.Dropdown(
                id="model-dropdown",
                options=[
                    {
                        "label": "Logistic Regression",
                        "value": "logistic_regression"
                    },
                    {
                        "label": "Random Forest",
                        "value": "random_forest"
                    }
                ],
                value="logistic_regression",
                clearable=False
            ),

            html.Br(),

            html.Label("Top Features"),

            dcc.Dropdown(
                id="topn-dropdown",
                options=[
                    {"label": "Top 5", "value": 5},
                    {"label": "Top 10", "value": 10},
                    {"label": "Top 15", "value": 15},
                    {"label": "Top 20", "value": 20}
                ],
                value=10,
                clearable=False
            ),

            html.Br(),

            dcc.Graph(
                id="shap-importance-chart"
            ),

            dash_table.DataTable(
                id="shap-table",
                page_size=20,
                style_table={
                    "overflowX": "auto"
                },
                style_cell={
                    "textAlign": "left",
                    "padding": "10px"
                },
                style_header={
                    "fontWeight": "bold"
                }
            )
        ]
    )