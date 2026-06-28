from pathlib import Path

import pandas as pd

from dash import (
    html,
    Input,
    Output,
)

from narrative import (
    generate_performance_summary,
)

from components.performance_tab import (
    build_metric_cards,
)

from visualisations.performance_plots import (
    performance_bar_chart,
)

from helpers.figure_helpers import (
    create_empty_figure,
)


def register_performance_callbacks(app):
    @app.callback(
        Output("performance-summary", "children"),
        Output("performance-metric-cards", "children"),
        Output("performance-comparison-chart", "figure"),
        Output("performance-comparison-table", "children"),
        Output("roc-curve-chart", "figure"),
        Input("analysis-status", "children"),
        Input("metric-selector", "value"),
    )
    def update_performance_tab(analysis_status, selected_metric):

        if analysis_status != "Analysis completed successfully.":
            return (
                "Run the analysis to generate a performance summary.",
                html.Div(),
                create_empty_figure("Upload a dataset and run analysis."),
                html.Div(),
                create_empty_figure("Upload a dataset and run analysis."),
            )
        file = Path("data/exports/evaluation_summary.csv")
        if not file.exists():
            return (
                "No evaluation results available.",
                html.Div(),
                create_empty_figure("No evaluation results."),
                html.Div(),
                create_empty_figure("No evaluation results."),
            )
        df = pd.read_csv(file)

        best_model = df.sort_values(by="f1_score", ascending=False).iloc[0]
        selected_metric = selected_metric or "accuracy"
        metric_titles = {
            "accuracy": "Accuracy Comparison",
            "precision": "Precision Comparison",
            "recall": "Recall Comparison",
            "f1_score": "F1 Score Comparison",
            "roc_auc": "ROC-AUC Comparison",
        }
        comparison_chart = performance_bar_chart(
            df, selected_metric, metric_titles[selected_metric]
        )
        table_df = df.sort_values(by="f1_score", ascending=False)
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
                                html.Th("ROC-AUC"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(row["model"].replace("_", " ").title()),
                                    html.Td(f"{row['accuracy']:.3f}"),
                                    html.Td(f"{row['precision']:.3f}"),
                                    html.Td(f"{row['recall']:.3f}"),
                                    html.Td(f"{row['f1_score']:.3f}"),
                                    html.Td(f"{row['roc_auc']:.3f}"),
                                ]
                            )
                            for _, row in table_df.iterrows()
                        ]
                    ),
                ],
                className="comparison-table",
            ),
            style={"overflowX": "auto"},
        )
        roc_chart = performance_bar_chart(df, "roc_auc", "ROC-AUC Comparison")

        return (
            generate_performance_summary(df),
            build_metric_cards(best_model),
            comparison_chart,
            comparison_table,
            roc_chart,
        )
