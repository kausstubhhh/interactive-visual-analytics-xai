from dash import dcc, html
from dash import dash_table


def create_decision_behaviour_tab():

    return html.Div(
        [

            html.H2(
                "Task 4: Decision Behaviour Analysis"
            ),

            html.Br(),

            html.Label("Dataset"),

            dcc.Dropdown(
                id="decision-dataset-dropdown",
                options=[
                    {
                        "label": "HELOC",
                        "value": "heloc"
                    },
                    {
                        "label": "BANK",
                        "value": "bank"
                    }
                ],
                value="heloc",
                clearable=False
            ),

            html.Br(),

            html.Label("Model"),

            dcc.Dropdown(
                id="decision-model-dropdown",
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
                id="decision-topn-dropdown",
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
                id="decision-chart"
            ),

            dash_table.DataTable(
                id="decision-table",
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