import pandas as pd
from pathlib import Path


def export_evaluation_summary(
    results,
    output_path
):
    """
    Export evaluation summary to CSV.

    Parameters
    ----------
    results : list[dict]

    output_path : str | Path
    """

    df = pd.DataFrame(results)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    return df