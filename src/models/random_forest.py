"""
Random Forest model definition.
"""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(random_state: int = 42):
    """
    Create a Random Forest classifier.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Configured classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    return model