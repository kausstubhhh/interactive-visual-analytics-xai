import io
import base64

import pandas as pd

from dash import (
    html,
    Input,
    Output,
    State,
)

from src.services.verification_service import (
    verify_dataset,
)

from src.services.dataset_service import (
    get_dataset_summary,
    get_dataset_profile,
)

from src.services.analysis_pipeline import (
    run_analysis_pipeline,
)


def register_dataset_callbacks(app):
    @app.callback(
        Output("upload-status", "children"),
        Output("analysis-status", "children", allow_duplicate=True),
        Output("dataset-summary", "children"),
        Output("dataset-preview-table", "data"),
        Output("dataset-preview-table", "columns"),
        Output("uploaded-dataset-store", "data"),
        Input("dataset-upload", "contents"),
        State("dataset-upload", "filename"),
        prevent_initial_call=True,
    )
    def upload_dataset(contents, filename):

        if contents is None:
            return ("No dataset uploaded.", "", "", [], [], None)

        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)

        if filename.endswith(".csv"):

            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):

            df = pd.read_excel(io.BytesIO(decoded))

        else:

            return ("Unsupported file type.", "", "", [], [], None)

        summary_data = get_dataset_summary(df)
        verification = verify_dataset(df)
        profile = get_dataset_profile(df)

        recommended = verification["compatible_targets"]

        recommended_text = ", ".join(recommended) if recommended else "None"

        summary = html.Div(
            [
                html.Div(
                    [
                        html.Div("Rows", className="metric-title"),
                        html.Div(f"{summary_data['rows']:,}", className="metric-value"),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div("Columns", className="metric-title"),
                        html.Div(
                            str(summary_data["columns"]), className="metric-value"
                        ),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div("Missing Values", className="metric-title"),
                        html.Div(
                            str(summary_data["missing_values"]),
                            className="metric-value",
                        ),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div("Numerical", className="metric-title"),
                        html.Div(
                            str(profile["numerical_columns"]), className="metric-value"
                        ),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div("Categorical", className="metric-title"),
                        html.Div(
                            str(profile["categorical_columns"]),
                            className="metric-value",
                        ),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div("Recommended Target", className="metric-title"),
                        html.Div(recommended_text, className="metric-value"),
                    ],
                    className="metric-card",
                ),
            ],
            className="metric-container",
        )

        preview = df.head(10)
        columns = [{"name": col, "id": col} for col in preview.columns]
        data = preview.to_dict("records")
        return (
            f"Uploaded: {filename}",
            "",
            summary,
            data,
            columns,
            df.to_json(orient="split"),
        )

    @app.callback(
        Output("target-column-dropdown", "options"),
        Output("target-column-dropdown", "value"),
        Input("uploaded-dataset-store", "data"),
    )
    def populate_target_dropdown(dataset_json):
        if dataset_json is None:
            return [], None

        df = pd.read_json(io.StringIO(dataset_json), orient="split")

        verification = verify_dataset(df)
        compatible = verification["compatible_targets"]
        options = []
        for column in df.columns:
            label = column
            if column in compatible:
                label = f"⭐ {column}"

            options.append({"label": label, "value": column})
        default_value = compatible[0] if compatible else df.columns[0]
        return (options, default_value)

    @app.callback(
        Output("run-analysis-button", "disabled"),
        Input("target-column-dropdown", "value"),
    )
    def enable_run_analysis(target_column):
        """
        Enable the analysis button only when
        a target has been selected.
        """

        return target_column is None

    @app.callback(
        Output("dataset-inspector-table", "data"),
        Output("dataset-inspector-table", "columns"),
        Input("uploaded-dataset-store", "data"),
    )
    def update_schema_summary(
        dataset_json,
    ):
        if dataset_json is None:
            return [], []
        df = pd.read_json(io.StringIO(dataset_json), orient="split")
        profile = get_dataset_profile(df)

        inspector_df = pd.DataFrame(profile["column_profiles"])

        inspector_df = inspector_df.rename(
            columns={
                "column": "Column",
                "type": "Type",
                "unique_count": "Values",
                "sample_values": "Example",
                "recommendation": "Recommendation",
            }
        )

        inspector_df["Example"] = inspector_df["Example"].apply(
            lambda values: ", ".join(map(str, values))
        )

        status_map = {"Compatible": "🟢 Compatible", "Feature": "⚪ Feature"}

        inspector_df["Recommendation"] = inspector_df["Recommendation"].map(status_map)

        inspector_columns = [
            {"name": column, "id": column} for column in inspector_df.columns
        ]

        inspector_data = inspector_df.to_dict("records")

        return (inspector_data, inspector_columns)

    @app.callback(
        Output("analysis-status", "children"),
        Input("run-analysis-button", "n_clicks"),
        State("uploaded-dataset-store", "data"),
        State("target-column-dropdown", "value"),
        prevent_initial_call=True,
    )
    def run_uploaded_analysis(n_clicks, dataset_json, target_column):

        try:

            if dataset_json is None:
                return "Upload a dataset first."

            if target_column is None:
                return "Select a target column."

            df = pd.read_json(io.StringIO(dataset_json), orient="split")

            run_analysis_pipeline(
                df=df, dataset_name="uploaded", target_column=target_column
            )
            return "Analysis completed successfully."

        except Exception as e:

            import traceback

            print("Pipeline execution failed.")

            traceback.print_exc()

            return f"Analysis failed: {str(e)}"
