from dash import dcc, html
from ui_components import create_workflow_banner
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

from components.dataset_management_tab import (
    create_dataset_management_tab
)

def create_layout():

    return html.Div(
        [
            dcc.Store(
                id="uploaded-dataset-store"
            ),

            dcc.Store(
                id="application-state",
                data={
                    "dataset_uploaded": False,
                    "target_selected": False,
                    "analysis_completed": False
                }
            ),
            
            html.Div(

                [

                    html.H1(
                        "Interactive Visual Analytics",
                        className="page-title"
                    ),

                    html.P(
                        (
                            "Explore, compare and explain "
                            "classification models using "
                            "interactive visual analytics."
                        ),
                        className="page-subtitle"
                    )

                ],

                className="dashboard-header"

            ),
            create_workflow_banner(),

            dcc.Tabs(
                id="main-tabs",
                value="dataset_management",
                className="dashboard-tabs",
                children=[
                    
                    dcc.Tab(
                        label="Dataset Management",
                        value="dataset_management",
                        children=[
                            create_dataset_management_tab()
                        ]
                    ),

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
        className="dashboard-container"
    )