import pandas as pd

from dash import (
    html,
    Input,
    Output,
)

from visualisations.decision_behaviour import (
    local_shap_chart,
)

from components.decision_behaviour_tab import (
    build_decision_metric_cards,
    generate_decision_summary,
)

from helpers.export_loader import (
    load_local_explanation_file,
)

from helpers.figure_helpers import (
    create_empty_figure,
)

def register_decision_behaviour_callbacks(app):
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
            
            dataset = "uploaded"

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