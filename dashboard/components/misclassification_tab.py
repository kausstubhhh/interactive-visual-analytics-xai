from dash import dcc, html, dash_table

from ui_components import (
    create_card,
    create_analysis_card,
    create_metric_card
)
def create_misclassification_tab():

    return html.Div(
        [

            html.H2(
                "Misclassification Analysis"
            ),

            html.P(
                (
                    "Investigate incorrect predictions made by each "
                    "classification model and understand where errors occur."
                )
            ),

            create_analysis_card(
                title="Misclassification Summary",
                component_id="misclassification-summary",
                icon="⚠️"
            ),

            html.Div(
                id="misclassification-metric-cards"
            ),

            html.Br(),

            create_card(

                "Analysis Controls",

                html.Div(
                    [
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
                        )
                    ]
                )
            ),

            create_card(
                "Confusion Matrix",
                dcc.Graph(
                    id="confusion-matrix-chart"
                )
            ),

            html.Br(),

            create_card(
                "Error Breakdown",
                dcc.Graph(
                    id="error-breakdown-chart"
                )
            ),
        ]
    )

def build_error_metric_cards(metrics):

    return html.Div(

        [

            create_metric_card(
                "False Positives",
                str(int(metrics["false_positives"]))
            ),

            create_metric_card(
                "False Negatives",
                str(int(metrics["false_negatives"]))
            ),

            create_metric_card(
                "Total Errors",
                str(int(metrics["total_errors"]))
            ),

            create_metric_card(
                "Error Rate",
                f"{metrics['error_rate']:.1%}"
            ),
            

        ],

        className="performance-summary-row"

    )