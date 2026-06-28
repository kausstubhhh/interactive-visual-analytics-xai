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

from narrative import generate_performance_summary

from visualisations.feature_importance import (
    feature_importance_chart
)

from visualisations.misclassification import (
    error_breakdown_figure
)

from visualisations.decision_behaviour import (
    local_shap_chart
)

from components.performance_tab import (
    performance_bar_chart,
    build_metric_cards
)
from components.feature_importance_tab import (
    build_feature_metric_cards,
    generate_feature_summary
)
from components.misclassification_tab import (
    build_error_metric_cards
)
from components.decision_behaviour_tab import (
    build_decision_metric_cards,
    generate_decision_summary
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
from src.services.verification_service import (
    verify_dataset
)

from src.services.dataset_service import (
    detect_dataset_schema,
    get_dataset_summary,
    get_dataset_profile
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

def generate_error_summary(metrics):

    return (

        f"The selected model produced "
        f"{int(metrics['total_errors'])} incorrect predictions "
        f"({metrics['error_rate']:.1%} error rate). "
        f"There were "
        f"{int(metrics['false_positives'])} false positives and "
        f"{int(metrics['false_negatives'])} false negatives."

    )

def register_callbacks(app):

    # --------------------------------------------------------
    # Task 2
    # Feature Importance Analysis
    # --------------------------------------------------------

    @app.callback(
        Output(
            "feature-summary",
            "children"
        ),

        Output(
            "feature-metric-cards",
            "children"
        ),

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
                "Run the analysis to generate a feature importance summary.",
                html.Div(),
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

        summary = generate_feature_summary(
            df,
            model
        )

        metric_cards = build_feature_metric_cards(
            df,
            model,
            top_n
        )

        print("=" * 60)
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print("=" * 60)
        
        figure = feature_importance_chart(
            df
        )

        table_columns = [
            {
                "name": column,
                "id": column
            }
            for column in df.columns
        ]

        df["importance"] = df["importance"].round(4)

        table_data = (
            df.to_dict(
                "records"
            )
        )

        return (
            summary,
            metric_cards,
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
            "misclassification-summary",
            "children"
        ),
        Output(
            "misclassification-metric-cards",
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

                "Run the analysis to investigate prediction errors.",
                html.Div(),
                create_empty_figure(
                    "Run analysis first"
                ),
                create_empty_figure(
                    "Run analysis first"
                ),
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
        figure = error_breakdown_figure(
            metrics
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
            generate_error_summary(metrics),
            build_error_metric_cards(metrics),
            confusion_figure,
            figure,
        )
    
    # --------------------------------------------------------
    # Task 4
    # Decision Behaviour Analysis
    # --------------------------------------------------------

    @app.callback(
            
        Output(
            "decision-summary",
            "children"
        ),
        Output(
            "decision-metric-cards",
            "children"
        ),
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
        ),
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

                "Run the analysis to generate a local explanation.",
                html.Div(),
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
        summary = generate_decision_summary(
            df,
            model
        )

        metric_cards = build_decision_metric_cards(
            df,
            model,
            top_n
        )
        figure = local_shap_chart(
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
            summary,
            metric_cards,
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
        
        summary_data = get_dataset_summary(df)
        verification = verify_dataset(df)
        profile = get_dataset_profile(df)

        recommended = verification["compatible_targets"]

        recommended_text = (
            ", ".join(recommended)
            if recommended
            else "None"
        )

        summary = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Rows",
                            className="metric-title"
                        ),
                        html.Div(
                            f"{summary_data['rows']:,}",
                            className="metric-value"
                        )
                    ],
                    className="metric-card"
                ),
                html.Div(
                    [
                        html.Div(
                            "Columns",
                            className="metric-title"
                        ),
                        html.Div(
                            str(summary_data["columns"]),
                            className="metric-value"
                        )
                    ],
                    className="metric-card"
                ),
                html.Div(
                    [
                        html.Div(
                            "Missing Values",
                            className="metric-title"
                        ),
                        html.Div(
                            str(summary_data["missing_values"]),
                            className="metric-value"
                        )
                    ],
                    className="metric-card"
                ),
                html.Div(
                    [
                        html.Div(
                            "Numerical",
                            className="metric-title"
                        ),
                        html.Div(
                            str(profile["numerical_columns"]),
                            className="metric-value"
                        )
                    ],
                    className="metric-card"
                ),
                html.Div(
                    [
                        html.Div(
                            "Categorical",
                            className="metric-title"
                        ),
                        html.Div(
                            str(profile["categorical_columns"]),
                            className="metric-value"
                        )
                    ],
                    className="metric-card"
                ),
                html.Div(
                    [
                        html.Div(
                            "Recommended Target",
                            className="metric-title"
                        ),

                        html.Div(
                            recommended_text,
                            className="metric-value"
                        )
                    ],
                    className="metric-card"
                )
            ],

            className="metric-container"
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
        if dataset_json is None:
            return [], None

        df = pd.read_json(
            io.StringIO(dataset_json),
            orient="split"
        )

        verification = verify_dataset(df)
        compatible = verification[
            "compatible_targets"
        ]
        options = []
        for column in df.columns:
            label = column
            if column in compatible:
                label = (
                    f"⭐ {column}"
                )

            options.append(
                {
                    "label": label,
                    "value": column
                }
            )
        default_value = (
            compatible[0]
            if compatible
            else df.columns[0]
        )
        return (
            options,
            default_value
        ) 
    

    @app.callback(
        Output(
            "run-analysis-button",
            "disabled"
        ),
        Input(
            "target-column-dropdown",
            "value"
        )
    )
    def enable_run_analysis(
        target_column
    ):
        """
        Enable the analysis button only when
        a target has been selected.
        """

        return target_column is None

    @app.callback(
        Output(
            "dataset-inspector-table",
            "data"
        ),

        Output(
            "dataset-inspector-table",
            "columns"
        ),
        Input(
            "uploaded-dataset-store",
            "data"
        ),
    )
    def update_schema_summary(
        dataset_json,
    ):
        if dataset_json is None:

            return [], []
        
        df = pd.read_json(
            io.StringIO(dataset_json),
            orient="split"
        )
        profile = get_dataset_profile(df)

        print("\nPROFILE:")
        print(profile)

        print("\nCOLUMN PROFILES:")
        print(profile["column_profiles"][:2])

        inspector_df = pd.DataFrame(
            profile["column_profiles"]
        )

        print("\nDATAFRAME COLUMNS:")
        print(inspector_df.columns.tolist())

        inspector_df = inspector_df.rename(
            columns={
                "column": "Column",
                "type": "Type",
                "unique_count": "Values",
                "sample_values": "Example",
                "recommendation": "Recommendation"
            }
        )

        inspector_df["Example"] = (
            inspector_df["Example"]
            .apply(
                lambda values: ", ".join(map(str, values))
            )
        )

        status_map = {
            "Compatible":
                "🟢 Compatible",
            "Feature":
                "⚪ Feature"
        }

        inspector_df["Recommendation"] = (
            inspector_df["Recommendation"]
            .map(status_map)
        )

        inspector_columns = [
            {
                "name": column,
                "id": column
            }
            for column in inspector_df.columns
        ]

        inspector_data = (
            inspector_df.to_dict("records")
        )

        return (
            inspector_data,
            inspector_columns
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
            "performance-summary",
            "children"
        ),
        Output(
            "performance-metric-cards",
            "children"
        ),
        Output(
            "performance-comparison-chart",
            "figure"
        ),
        Output(
            "performance-comparison-table",
            "children"
        ),
        Output(
            "roc-curve-chart",
            "figure"
        ),
        Input(
            "analysis-status",
            "children"
        ),
        Input(
            "metric-selector",
            "value"
        )
    )
    def update_performance_tab(
        analysis_status,
        selected_metric
    ):
        
        print("=" * 60)
        print("PERFORMANCE CALLBACK")
        print("analysis_status:", analysis_status)
        print("selected_metric:", selected_metric)
        print("=" * 60)

        if analysis_status != "Analysis completed successfully.":
            return (
                "Run the analysis to generate a performance summary.",
                html.Div(),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
                html.Div(),
                create_empty_figure(
                    "Upload a dataset and run analysis."
                ),
            )
        file = Path(
            "data/exports/evaluation_summary.csv"
        )
        if not file.exists():
            return (
                "No evaluation results available.",
                html.Div(),
                create_empty_figure(
                    "No evaluation results."
                ),
                html.Div(),
                create_empty_figure(
                    "No evaluation results."
                ),
            )
        df = pd.read_csv(file)
        table = df.copy()

        columns = [
            {
                "name": c,
                "id": c
            }
            for c in table.columns
        ]
        best_model = (
            df.sort_values(
                by="f1_score",
                ascending=False
            )
            .iloc[0]
        )
        selected_metric = selected_metric or "accuracy"
        metric_titles = {
            "accuracy": "Accuracy Comparison",
            "precision": "Precision Comparison",
            "recall": "Recall Comparison",
            "f1_score": "F1 Score Comparison",
            "roc_auc": "ROC-AUC Comparison"
        }
        comparison_chart = performance_bar_chart(
            df,
            selected_metric,
            metric_titles[selected_metric]
        )
        table_df = df.sort_values(
            by="f1_score",
            ascending=False
        )
        comparison_table = html.Div(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Model"),
                                html.Th("Accuracy"),
                                html.Th("Precision"),
                                html.Th("Recall"),
                                html.Th("F1"),
                                html.Th("ROC-AUC")
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(
                                        row["model"]
                                        .replace("_", " ")
                                        .title()
                                    ),
                                    html.Td(
                                        f"{row['accuracy']:.3f}"
                                    ),
                                    html.Td(
                                        f"{row['precision']:.3f}"
                                    ),
                                    html.Td(
                                        f"{row['recall']:.3f}"
                                    ),
                                    html.Td(
                                        f"{row['f1_score']:.3f}"
                                    ),
                                    html.Td(
                                        f"{row['roc_auc']:.3f}"
                                    )
                                ]
                            )
                            for _, row in table_df.iterrows()
                        ]
                    )
                ],
                className="comparison-table"
            ),
            style={
                "overflowX": "auto"
            }
        )
        roc_chart = performance_bar_chart(
            df,
            "roc_auc",
            "ROC-AUC Comparison"
        )
        models = [
            "logistic_regression",
            "random_forest"
        ]

    
        return (
            generate_performance_summary(df),
            build_metric_cards(best_model),
            comparison_chart,
            comparison_table,
            roc_chart
        )