from dash import dcc, html


def create_misclassification_tab():

    return html.Div(
        [

            html.H2(
                "Task 3: Misclassification Analysis"
            ),

            html.Br(),

            html.Label("Dataset"),

            dcc.Dropdown(
                id="error-dataset-dropdown",
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
                id="error-model-dropdown",
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

            html.Div(
                [

                    html.Div(
                        [
                            html.H4("False Positives"),
                            html.H3(id="false-positive-card")
                        ],
                        style={
                            "width": "24%",
                            "display": "inline-block",
                            "textAlign": "center"
                        }
                    ),

                    html.Div(
                        [
                            html.H4("False Negatives"),
                            html.H3(id="false-negative-card")
                        ],
                        style={
                            "width": "24%",
                            "display": "inline-block",
                            "textAlign": "center"
                        }
                    ),

                    html.Div(
                        [
                            html.H4("Total Errors"),
                            html.H3(id="total-errors-card")
                        ],
                        style={
                            "width": "24%",
                            "display": "inline-block",
                            "textAlign": "center"
                        }
                    ),

                    html.Div(
                        [
                            html.H4("Error Rate"),
                            html.H3(id="error-rate-card")
                        ],
                        style={
                            "width": "24%",
                            "display": "inline-block",
                            "textAlign": "center"
                        }
                    )
                ]
            ),

            html.Br(),

            dcc.Graph(
                id="error-breakdown-chart"
            )
        ]
    )