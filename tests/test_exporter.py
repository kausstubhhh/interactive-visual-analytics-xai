from pathlib import Path

from src.evaluation.exporter import (
    export_evaluation_summary
)


def test_export_summary(tmp_path):

    data = [
        {
            "dataset": "TEST",
            "model": "LR",
            "accuracy": 0.8
        }
    ]

    output_file = (
        tmp_path /
        "summary.csv"
    )

    export_evaluation_summary(
        data,
        output_file
    )

    assert output_file.exists()