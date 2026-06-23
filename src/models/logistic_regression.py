"""
Logistic Regression model definition.
"""

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(random_state: int = 42):
    """
    Create a Logistic Regression classifier.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Configured classifier.
    """

    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver="lbfgs"
    )

    return model