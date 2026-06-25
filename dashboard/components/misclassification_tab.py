from dash import dcc, html


def create_misclassification_tab():

    return html.Div(
        [

            html.H2(
                "Task 3: Misclassification Analysis"
            ),

            html.H4(
                "Question"
            ),

            html.P(
                "Where does the selected model make incorrect predictions?"
            ),

            html.Br(),

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

            html.Br(),

            html.Div(
                [

                    # False Positives
                    html.Div(
                        [
                            html.H4("False Positives"),
                            html.Small(
                                "Predicted Positive\nActually Negative"
                            ),
                            html.Br(),
                            html.Br(),
                            html.H3(id="false-positive-card")
                        ],
                        style={
                            "flex": "1",
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "backgroundColor": "#FCECEC"
                        }
                    ),

                    # False Negatives
                    html.Div(
                        [
                            html.H4("False Negatives"),
                            html.Small(
                                "Predicted Negative\nActually Positive"
                            ),
                            html.Br(),
                            html.Br(),
                            html.H3(id="false-negative-card")
                        ],
                        style={
                            "flex": "1",
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "backgroundColor": "#FCECEC"
                        }
                    ),

                    # Total Errors
                    html.Div(
                        [
                            html.H4("Total Errors"),
                            html.Small(
                                "False Positives + False Negatives"
                            ),
                            html.Br(),
                            html.Br(),
                            html.H3(id="total-errors-card")
                        ],
                        style={
                            "flex": "1",
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "backgroundColor": "#FFF8DC"
                        }
                    ),

                    # Error Rate
                    html.Div(
                        [
                            html.H4("Error Rate"),
                            html.Small(
                                "Errors / Total Predictions"
                            ),
                            html.Br(),
                            html.Br(),
                            html.H3(id="error-rate-card")
                        ],
                        style={
                            "flex": "1",
                            "padding": "20px",
                            "border": "1px solid #ddd",
                            "borderRadius": "8px",
                            "textAlign": "center",
                            "backgroundColor": "#EEF7FF"
                        }
                    )

                ],
                style={
                    "display": "flex",
                    "gap": "15px",
                    "marginTop": "20px",
                    "marginBottom": "20px"
                }
            ),

            html.Br(),

            html.H3(
                "Confusion Matrix"
            ),

            dcc.Graph(
                id="confusion-matrix-chart"
            ),

            html.Br(),

            html.H3(
                "Error Breakdown"
            ),

            dcc.Graph(
                id="error-breakdown-chart"
            ),
        ]
    )