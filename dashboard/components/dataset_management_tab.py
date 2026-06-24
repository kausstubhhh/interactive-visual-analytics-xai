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

            html.Hr(),

            html.H3(
                "1. Upload Dataset"
            ),

            dcc.Upload(
                id="dataset-upload",
                children=html.Div(
                    [
                        "Drag and Drop or ",
                        html.A("Select CSV File")
                    ]
                ),
                style={
                    "width": "100%",
                    "height": "80px",
                    "lineHeight": "80px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "marginBottom": "20px"
                },
                multiple=False
            ),

            html.Div(
                id="upload-status",
                children="No dataset uploaded."
            ),

            html.Hr(),

            html.H3(
                "2. Dataset Summary"
            ),

            html.Div(
                id="dataset-summary"
            ),

            html.Hr(),

            html.H3(
                "3. Target Selection"
            ),

            dcc.Dropdown(
                id="target-column-dropdown",
                options=[],
                placeholder=(
                    "Upload a dataset first"
                ),
                clearable=False
            ),

            html.Hr(),

            html.Br(),

            html.Button(
                "Run Analysis",
                id="run-analysis-button",
                n_clicks=0,
                style={
                    "padding": "12px",
                    "fontSize": "16px"
                }
            ),

            html.Br(),
            html.Br(),

            html.Div(
                id="analysis-status"
            ),

            html.H3(
                "4. Schema Detection"
            ),

            html.Div(
                id="schema-summary"
            ),

            html.Hr(),

            html.H3(
                "5. Dataset Preview"
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
        ]
    )