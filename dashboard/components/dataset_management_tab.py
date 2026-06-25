from dash import dash_table
from dash import dcc
from dash import html


def create_dataset_management_tab():

    return html.Div(
        [

            html.H2(
                "Dataset Management"
            ),

            html.P(
                (
                    "Upload a classification dataset to "
                    "compare models, investigate feature "
                    "importance, analyse errors, and "
                    "understand model decision behaviour."
                )
            ),

            # -----------------------------
            # Upload Card
            # -----------------------------

            html.Div(

                [

                    html.H3(
                        "Upload Dataset"
                    ),

                    dcc.Upload(
                        id="dataset-upload",
                        children=html.Div(
                            [
                                "Drag and Drop or ",
                                html.A("Select CSV File")
                            ]
                        ),
                        className="upload-box",
                        multiple=False
                    ),

                    html.Div(
                        id="upload-status",
                        children="No dataset uploaded."
                    )

                ],

                className="dashboard-card"

            ),

            # -----------------------------
            # Summary Card
            # -----------------------------

            html.Div(

                [

                    html.H3(
                        "Dataset Summary"
                    ),

                    html.Div(
                        id="dataset-summary"
                    )

                ],

                className="dashboard-card"

            ),

            # -----------------------------
            # Target Selection Card
            # -----------------------------

            html.Div(

                [

                    html.H3(
                        "Target Selection"
                    ),

                    dcc.Dropdown(
                        id="target-column-dropdown",
                        options=[],
                        placeholder="Upload a dataset first",
                        clearable=False
                    ),

                    html.Br(),

                    html.Button(
                        "Run Analysis",
                        id="run-analysis-button",
                        n_clicks=0
                    ),

                    html.Br(),
                    html.Br(),

                    html.Div(
                        id="analysis-status"
                    )

                ],

                className="dashboard-card"

            ),

            # -----------------------------
            # Schema Card
            # -----------------------------

            html.Div(

                [

                    html.H3(
                        "Detected Schema"
                    ),

                    html.Div(
                        id="schema-summary"
                    )

                ],

                className="dashboard-card"

            ),

            # -----------------------------
            # Preview Card
            # -----------------------------

            html.Div(

                [

                    html.H3(
                        "Dataset Preview"
                    ),

                    dash_table.DataTable(
                        id="dataset-preview-table",
                        page_size=10,
                        style_table={
                            "overflowX": "auto"
                        },
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px"
                        }
                    )

                ],

                className="dashboard-card"

            )

        ]
    )