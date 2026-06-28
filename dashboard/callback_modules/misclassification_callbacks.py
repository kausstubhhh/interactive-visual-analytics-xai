from dash import (
    html,
    Input,
    Output,
)

from visualisations.misclassification import (
    error_breakdown_figure,
)

from components.misclassification_tab import (
    build_error_metric_cards,
)

from helpers.export_loader import (
    load_error_file,
    load_confusion_file,
)

from helpers.figure_helpers import (
    create_empty_figure,
    create_confusion_figure,
)

from helpers.summary_helpers import (
    generate_error_summary,
)

def register_misclassification_callbacks(app):
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