from dash import (
    html,
    Input,
    Output,
)

from visualisations.feature_importance import (
    feature_importance_chart,
)

from components.feature_importance_tab import (
    build_feature_metric_cards,
    generate_feature_summary,
)

from helpers.export_loader import (
    load_shap_file,
)

from helpers.figure_helpers import (
    create_empty_figure,
)


def register_feature_importance_callbacks(app):
    @app.callback(
        Output("feature-summary", "children"),
        Output("feature-metric-cards", "children"),
        Output("shap-importance-chart", "figure"),
        Output("shap-table", "data"),
        Output("shap-table", "columns"),
        Input("model-dropdown", "value"),
        Input("topn-dropdown", "value"),
        Input("analysis-status", "children"),
    )
    def update_feature_importance(model, top_n, analysis_status):

        if analysis_status != ("Analysis completed successfully."):
            return (
                "Run the analysis to generate a feature importance summary.",
                html.Div(),
                create_empty_figure("Upload a dataset and run analysis."),
                [],
                [],
            )

        dataset = "uploaded"

        df = load_shap_file(dataset, model)

        df = (
            df.sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        df["feature"] = (
            df["feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )
        df.insert(0, "rank", range(1, len(df) + 1))

        summary = generate_feature_summary(df, model)

        metric_cards = build_feature_metric_cards(df, model, top_n)

        figure = feature_importance_chart(df)

        table_columns = [{"name": column, "id": column} for column in df.columns]

        df["importance"] = df["importance"].round(4)

        table_data = df.to_dict("records")

        return (summary, metric_cards, figure, table_data, table_columns)
