from dash import dcc, html

from components.performance_tab import (
    create_performance_tab
)

from components.feature_importance_tab import (
    create_feature_importance_tab
)

from components.misclassification_tab import (
    create_misclassification_tab
)

from components.decision_behaviour_tab import (
    create_decision_behaviour_tab
)


def create_layout():

    return html.Div(
        [
            html.H1(
                "Interactive Visual Analytics Dashboard"
            ),

            dcc.Tabs(
                id="main-tabs",
                value="performance",
                children=[

                    dcc.Tab(
                        label="Performance Comparison",
                        value="performance",
                        children=[
                            create_performance_tab()
                        ]
                    ),

                    dcc.Tab(
                        label="Feature Importance",
                        value="feature_importance",
                        children=[
                            create_feature_importance_tab()
                        ]
                    ),

                    dcc.Tab(
                        label="Misclassification",
                        value="misclassification",
                        children=[
                            create_misclassification_tab()
                        ]
                    ),

                    dcc.Tab(
                        label="Decision Behaviour",
                        value="decision_behaviour",
                        children=[
                            create_decision_behaviour_tab()
                        ]
                    )
                ]
            )
        ],
        style={
            "padding": "20px"
        }
    )