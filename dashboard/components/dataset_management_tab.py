from dash import dash_table
from dash import dcc
from dash import html

from ui_components import create_card


def create_dataset_management_tab():

    return html.Div(
        [
            html.H2("Dataset Management"),
            html.P(
                (
                    "Upload a classification dataset to "
                    "compare models, investigate feature "
                    "importance, analyse errors, and "
                    "understand model decision behaviour."
                )
            ),
            # -----------------------------------
            # Upload Dataset
            # -----------------------------------
            create_card(
                "Upload Dataset",
                dcc.Upload(
                    id="dataset-upload",
                    children=html.Div(["Drag and Drop or ", html.A("Select CSV File")]),
                    className="upload-box",
                    multiple=False,
                ),
                html.Div(id="upload-status", children="No dataset uploaded."),
            ),
            # -----------------------------------
            # Dataset Summary
            # -----------------------------------
            create_card("Dataset Summary", html.Div(id="dataset-summary")),
            # -----------------------------------
            # Target Selection
            # -----------------------------------
            create_card(
                "Target Selection",
                dcc.Dropdown(
                    id="target-column-dropdown",
                    options=[],
                    placeholder="Upload a dataset first",
                    clearable=False,
                ),
                html.Br(),
                html.Button(
                    "Run Analysis",
                    id="run-analysis-button",
                    n_clicks=0,
                    disabled=True,
                    className="primary-button",
                ),
                html.Br(),
                html.Br(),
                html.Div(id="analysis-status"),
            ),
            # -----------------------------------
            # Schema Detection
            # -----------------------------------
            create_card(
                "Dataset Inspector",
                dash_table.DataTable(
                    id="dataset-inspector-table",
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "fontFamily": "Inter",
                    },
                    style_header={"fontWeight": "bold", "backgroundColor": "#F6F8FC"},
                ),
            ),
            # -----------------------------------
            # Dataset Preview
            # -----------------------------------
            create_card(
                "Dataset Preview",
                dash_table.DataTable(
                    id="dataset-preview-table",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px"},
                ),
            ),
        ]
    )
