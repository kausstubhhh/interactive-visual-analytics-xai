from pathlib import Path
import sys
import io
import base64

import pandas as pd
import plotly.express as px

from dash import (
    html,
    Input,
    Output,
    State
)

# Project Root

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )

# Services

from src.services.dataset_service import (
    detect_dataset_schema
)

from src.services.analysis_pipeline import (
    run_analysis_pipeline
)
# ============================================================
# PATHS
# ============================================================

SHAP_EXPORT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "exports"
    / "shap"
)

ERROR_EXPORT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "exports"
    / "errors"
)

LOCAL_EXPLANATION_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "exports"
    / "local_explanations"
)

# ============================================================
# SHAP FUNCTIONS
# ============================================================

def load_shap_file(
    dataset: str,
    model: str
) -> pd.DataFrame:
    """
    Load SHAP importance CSV.
    """

    filename = (
        f"{dataset}_{model}_importance.csv"
    )

    file_path = (
        SHAP_EXPORT_DIR / filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"SHAP file not found: {file_path}"
        )

    return pd.read_csv(file_path)


def create_shap_figure(
    df: pd.DataFrame
):
    """
    Create horizontal SHAP importance chart.
    """

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance (SHAP)"
    )

    fig.update_layout(
        height=max(
            700,
            len(df) * 35
        ),
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    return fig


# ============================================================
# ERROR FUNCTIONS
# ============================================================

def load_error_file(
    dataset: str,
    model: str
) -> pd.DataFrame:
    """
    Load misclassification summary CSV.
    """

    filename = (
        f"{dataset}_{model}_errors.csv"
    )

    file_path = (
        ERROR_EXPORT_DIR / filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Error file not found: {file_path}"
        )

    return pd.read_csv(file_path)


def create_error_figure(
    df: pd.DataFrame
):
    """
    Create error breakdown chart.
    """

    chart_df = df[
        df["metric"].isin(
            [
                "false_positives",
                "false_negatives",
                "total_errors"
            ]
        )
    ]

    fig = px.bar(
        chart_df,
        x="metric",
        y="value",
        text="value",
        title="Error Breakdown"
    )

    fig.update_layout(
        height=500
    )

    return fig


# ============================================================
# CALLBACKS
# ============================================================
def load_local_explanation_file(
    dataset: str,
    model: str
) -> pd.DataFrame:
    """
    Load local explanation CSV.
    """

    filename = (
        f"{dataset}_{model}_instance_0.csv"
    )

    file_path = (
        LOCAL_EXPLANATION_DIR
        / filename
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Local explanation file not found: {file_path}"
        )

    return pd.read_csv(file_path)


def create_decision_figure(
    df: pd.DataFrame
):
    """
    Create local explanation chart.
    """

    fig = px.bar(
        df,
        x="contribution",
        y="feature",
        orientation="h",
        color="contribution",
        title="Local Feature Contributions"
    )

    fig.update_layout(
        height=700,
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    return fig

def register_callbacks(app):

    # --------------------------------------------------------
    # Task 2
    # Feature Importance Analysis
    # --------------------------------------------------------

    @app.callback(
        Output(
            "shap-importance-chart",
            "figure"
        ),
        Output(
            "shap-table",
            "data"
        ),
        Output(
            "shap-table",
            "columns"
        ),
        Input(
            "dataset-dropdown",
            "value"
        ),
        Input(
            "model-dropdown",
            "value"
        ),
        Input(
            "topn-dropdown",
            "value"
        )
    )
    def update_feature_importance(
        dataset,
        model,
        top_n
    ):

        df = load_shap_file(
            dataset,
            model
        )

        df = (
            df.sort_values(
                "importance",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )
        df["feature"] = (
            df["feature"]
            .str.replace(
                "numerical__",
                "",
                regex=False
            )
            .str.replace(
                "categorical__",
                "",
                regex=False
            )
        )
        df.insert(
            0,
            "rank",
            range(
                1,
                len(df) + 1
            )
        )

        figure = create_shap_figure(
            df
        )

        table_columns = [
            {
                "name": column,
                "id": column
            }
            for column in df.columns
        ]

        table_data = (
            df.to_dict(
                "records"
            )
        )

        return (
            figure,
            table_data,
            table_columns
        )

    # --------------------------------------------------------
    # Task 3
    # Misclassification Analysis
    # --------------------------------------------------------

    @app.callback(
        Output(
            "false-positive-card",
            "children"
        ),
        Output(
            "false-negative-card",
            "children"
        ),
        Output(
            "total-errors-card",
            "children"
        ),
        Output(
            "error-rate-card",
            "children"
        ),
        Output(
            "error-breakdown-chart",
            "figure"
        ),
        Input(
            "error-dataset-dropdown",
            "value"
        ),
        Input(
            "error-model-dropdown",
            "value"
        )
    )
    def update_error_analysis(
        dataset,
        model
    ):

        df = load_error_file(
            dataset,
            model
        )

        metrics = dict(
            zip(
                df["metric"],
                df["value"]
            )
        )

        figure = create_error_figure(
            df
        )

        return (
            f"{int(metrics['false_positives'])}",
            f"{int(metrics['false_negatives'])}",
            f"{int(metrics['total_errors'])}",
            f"{metrics['error_rate']:.3f}",
            figure
        )
    
    # --------------------------------------------------------
    # Task 4
    # Decision Behaviour Analysis
    # --------------------------------------------------------

    @app.callback(
        Output(
            "decision-chart",
            "figure"
        ),
        Output(
            "decision-table",
            "data"
        ),
        Output(
            "decision-table",
            "columns"
        ),
        Input(
            "decision-dataset-dropdown",
            "value"
        ),
        Input(
            "decision-model-dropdown",
            "value"
        ),
        Input(
            "decision-topn-dropdown",
            "value"
        )
    )
    
    def update_decision_behaviour(
        dataset,
        model,
        top_n
    ):

        df = load_local_explanation_file(
            dataset,
            model
        )

        df = (
            df.sort_values(
                "abs_contribution",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        df["feature"] = (
            df["feature"]
            .str.replace(
                "numerical__",
                "",
                regex=False
            )
            .str.replace(
                "categorical__",
                "",
                regex=False
            )
        )
        
        df.insert(
            0,
            "rank",
            range(
                1,
                len(df) + 1
            )
        )

        figure = create_decision_figure(
            df
        )

        table_columns = [
            {
                "name": column,
                "id": column
            }
            for column in df.columns
        ]

        table_data = (
            df.to_dict(
                "records"
            )
        )

        return (
            figure,
            table_data,
            table_columns
        )
    
    @app.callback(
        Output(
            "upload-status",
            "children"
        ),
        Output(
            "dataset-summary",
            "children"
        ),
        Output(
            "dataset-preview-table",
            "data"
        ),
        Output(
            "dataset-preview-table",
            "columns"
        ),
        Output(
            "uploaded-dataset-store",
            "data"
        ),
        Input(
            "dataset-upload",
            "contents"
        ),
        State(
            "dataset-upload",
            "filename"
        ),
        prevent_initial_call=True
    )
    def upload_dataset(
        contents,
        filename
    ):

        if contents is None:
            return (
                "No dataset uploaded.",
                "",
                [],
                [],
                None
            )

        content_type, content_string = (
            contents.split(",")
        )

        decoded = base64.b64decode(
            content_string
        )

        if filename.endswith(".csv"):

            df = pd.read_csv(
                io.StringIO(
                    decoded.decode("utf-8")
                )
            )

        elif (
            filename.endswith(".xlsx")
            or filename.endswith(".xls")
        ):

            df = pd.read_excel(
                io.BytesIO(decoded)
            )

        else:

            return (
                "Unsupported file type.",
                "",
                [],
                [],
                None
            )

        summary = html.Div(
            [
                html.P(
                    f"Rows: {len(df)}"
                ),

                html.P(
                    f"Columns: {len(df.columns)}"
                ),

                html.P(
                    f"Missing Values: "
                    f"{int(df.isna().sum().sum())}"
                )
            ]
        )

        preview = df.head(10)

        columns = [
            {
                "name": col,
                "id": col
            }
            for col in preview.columns
        ]

        data = preview.to_dict(
            "records"
        )

        return (
            f"Uploaded: {filename}",
            summary,
            data,
            columns,
            df.to_json(
                orient="split"
            )
        )
    
    @app.callback(
        Output(
            "target-column-dropdown",
            "options"
        ),
        Output(
            "target-column-dropdown",
            "value"
        ),
        Input(
            "uploaded-dataset-store",
            "data"
        )
    )
    def populate_target_dropdown(
        dataset_json
    ):
        try:

            if dataset_json is None:

                return [], None

            df = pd.read_json(
                io.StringIO(dataset_json),
                orient="split"
            )

            options = [
                {
                    "label": column,
                    "value": column
                }
                for column in df.columns
            ]

            default_target = (
                df.columns[0]
            )

            return (
                options,
                default_target
            )
        except Exception as e:
            print(e)
            raise 
    
    @app.callback(
        Output(
            "schema-summary",
            "children"
        ),
        Input(
            "uploaded-dataset-store",
            "data"
        ),
        Input(
            "target-column-dropdown",
            "value"
        )
    )
    def update_schema_summary(
        dataset_json,
        target_column
    ):

        if (
            dataset_json is None
            or target_column is None
        ):
            return ""

        df = pd.read_json(
            io.StringIO(dataset_json),
            orient="split"
        )

        schema = (
            detect_dataset_schema(
                df,
                target_column
            )
        )

        return html.Div(
            [

                html.P(
                    f"Target: "
                    f"{schema['target']}"
                ),

                html.P(
                    f"Categorical Features: "
                    f"{len(schema['categorical_columns'])}"
                ),

                html.P(
                    f"Numerical Features: "
                    f"{len(schema['numerical_columns'])}"
                ),

                html.H4(
                    "Categorical Columns"
                ),

                html.Ul(
                    [
                        html.Li(col)
                        for col in
                        schema[
                            "categorical_columns"
                        ]
                    ]
                ),

                html.H4(
                    "Numerical Columns"
                ),

                html.Ul(
                    [
                        html.Li(col)
                        for col in
                        schema[
                            "numerical_columns"
                        ]
                    ]
                )
            ]
        )
    @app.callback(
        Output(
            "analysis-status",
            "children"
        ),
        Input(
            "run-analysis-button",
            "n_clicks"
        ),
        State(
            "uploaded-dataset-store",
            "data"
        ),
        State(
            "target-column-dropdown",
            "value"
        ),
        prevent_initial_call=True
    )
    def run_uploaded_analysis(
        n_clicks,
        dataset_json,
        target_column
    ):

        print("\n")
        print("=" * 60)
        print("RUN ANALYSIS CLICKED")
        print("=" * 60)

        try:

            if dataset_json is None:

                print("ERROR: dataset_json is None")

                return (
                    "Upload a dataset first."
                )

            if target_column is None:

                print("ERROR: target column is None")

                return (
                    "Select a target column."
                )

            print(
                f"Target Column: {target_column}"
            )

            df = pd.read_json(
                io.StringIO(dataset_json),
                orient="split"
            )

            print(
                f"Dataset Shape: {df.shape}"
            )

            print(
                "Starting Analysis Pipeline..."
            )

            results = run_analysis_pipeline(
                df=df,
                dataset_name="uploaded",
                target_column=target_column
            )

            print(
                "Pipeline Completed Successfully"
            )

            print(
                "=" * 60
            )

            return (
                "Analysis completed successfully."
            )

        except Exception as e:

            import traceback

            print(
                "\nPIPELINE FAILED\n"
            )

            traceback.print_exc()

            return (
                f"Analysis failed: {str(e)}"
            )