from pathlib import Path
import sys
import io
import base64
import plotly.graph_objects as go

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

from components.performance_tab import (
    create_metric_figure
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
    print(df)
    print(df.empty)
    print(df.columns.tolist())
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

def load_confusion_file(
    dataset,
    model
):
    """
    Load confusion matrix CSV.
    """

    filename = (
        f"{dataset}_{model}_confusion.csv"
    )

    file_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "exports"
        / "confusion"
        / filename
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Confusion file not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        index_col=0
    )

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
            ]
        )
    ]
    chart_df = chart_df.copy()

    chart_df["metric"] = chart_df["metric"].replace(
    {
        "false_positives": "False Positives",
        "false_negatives": "False Negatives"
    }
)

    print(df)
    print(df.dtypes)
    print(type(df))

    fig = px.bar(
        chart_df,
        x="metric",
        y="value",
        text="value",
        title="Error Breakdown"
    )

    fig.update_layout(
        title="Error Breakdown",
        xaxis_title="Error Type",
        yaxis_title="Count",
        xaxis_tickangle=0,
        height=500
    )

    return fig

def create_confusion_figure(df):
    """
    Create confusion matrix heatmap.
    """

    # Rearrange matrix:
    # [[TP, FN],
    #  [FP, TN]]

    matrix = [
        [df.iloc[1, 1], df.iloc[1, 0]],
        [df.iloc[0, 1], df.iloc[0, 0]]
    ]

    labels = [
        ["TP", "FN"],
        ["FP", "TN"]
    ]

    annotations = [
        [
            f"{labels[i][j]}<br>{matrix[i][j]}"
            for j in range(2)
        ]
        for i in range(2)
    ]

    fig = px.imshow(
        matrix,
        x=[
            "Positive",
            "Negative"
        ],
        y=[
            "Positive",
            "Negative"
        ],
        color_continuous_scale="Blues",
        text_auto=False,
        aspect="equal"
    )

    fig.update_traces(
        text=annotations,
        texttemplate="%{text}"
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=550
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

def create_empty_figure(message):
    """
    Create an empty Plotly figure with a centered message.
    """

    fig = go.Figure()

    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 18
                }
            }
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450
    )

    return fig

def create_decision_figure(
    df: pd.DataFrame
):
    """
    Create local explanation chart.
    """

    print(df)
    print(df.empty)
    print(df.columns.tolist())
    print(df.dtypes)
    print(type(df))

    fig = px.bar(
        df,
        x="SHAP Value",
        y="feature",
        orientation="h",
        color="SHAP Value",
        text="SHAP Value",
        title="Local Feature Contributions"
    )

    fig.update_layout(
        title="Local Feature Contributions",
        height=500
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
            "model-dropdown",
            "value"
        ),
        Input(
            "topn-dropdown",
            "value"
        ),
        Input(
            "analysis-status",
            "children"
        )
    )
    def update_feature_importance(
        model,
        top_n,
        analysis_status
    ):
        if analysis_status != (
            "Analysis completed successfully."
        ):
            return (
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                [],
                []
            )
        
        dataset ="uploaded"

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

        print("=" * 60)
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print("=" * 60)
        
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
            "confusion-matrix-chart",
            "figure"
        ),
        Output(
            "error-breakdown-chart",
            "figure"
        ),
        Input(
            "error-model-dropdown",
            "value"
        ),
        Input(
            "analysis-status",
            "children"
        )
    )
    def update_error_analysis(
        model,
        analysis_status
    ):
        if analysis_status != (
            "Analysis completed successfully."
        ):
            return (
                "",
                "",
                "",
                "",
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                )
            )
            
        
        dataset = "uploaded"

        # Load error summary
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

        # Create error chart
        figure = create_error_figure(
            df
        )

        # Load confusion matrix
        confusion_df = load_confusion_file(
            dataset,
            model
        )

        confusion_figure = create_confusion_figure(
            confusion_df
        )

        return (
            f"{int(metrics['false_positives'])}",
            f"{int(metrics['false_negatives'])}",
            f"{int(metrics['total_errors'])}",
            f"{metrics['error_rate']:.3f}",
            confusion_figure,
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
            "analysis-status",
            "children"
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
        analysis_status,
        model,
        top_n,
    ):
        if analysis_status != (
            "Analysis completed successfully."
        ):
            return (
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                [],
                []
            )
        
        dataset ="uploaded"

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
        df["contribution"] = (
            df["contribution"]
            .round(3)
        )
        
        df.insert(
            0,
            "rank",
            range(
                1,
                len(df) + 1
            )
        )

        df = df.drop(
            columns=[
                "abs_contribution"
            ]
        )

        df = df.rename(
            columns={
                "contribution": "SHAP Value"
            }
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
            "analysis-status",
            "children",
            allow_duplicate=True
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
            "",
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
        
    @app.callback(
        Output(
            "accuracy-chart",
            "figure"
        ),
        Output(
            "precision-chart",
            "figure"
        ),
        Output(
            "recall-chart",
            "figure"
        ),
        Output(
            "f1-chart",
            "figure"
        ),
        Output(
            "roc-chart",
            "figure"
        ),
        Input(
            "analysis-status",
            "children"
        )
    )
    
    def update_performance_tab(
        analysis_status
    ):

        if analysis_status != (
            "Analysis completed successfully."
        ):
            return (
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                )
            )

        file = Path(
            "data/exports/evaluation_summary.csv"
        )

        if not file.exists():
            return (
                create_empty_figure(
                    "No evaluation results."
                ),
                create_empty_figure(
                    "No evaluation results."
                ),
                create_empty_figure(
                    "No evaluation results."
                ),
                create_empty_figure(
                    "No evaluation results."
                ),
                create_empty_figure(
                    "No evaluation results."
                )
            )
        

        df = pd.read_csv(file)

        return (

            create_metric_figure(
                df,
                "accuracy",
                "Accuracy Comparison"
            ),

            create_metric_figure(
                df,
                "precision",
                "Precision Comparison"
            ),

            create_metric_figure(
                df,
                "recall",
                "Recall Comparison"
            ),

            create_metric_figure(
                df,
                "f1_score",
                "F1 Score Comparison"
            ),

            create_metric_figure(
                df,
                "roc_auc",
                "ROC-AUC Comparison"
            )
        )