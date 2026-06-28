from callback_modules.feature_importance_callbacks import (
    register_feature_importance_callbacks,
)

from callback_modules.misclassification_callbacks import (
    register_misclassification_callbacks,
)

from callback_modules.decision_behaviour_callbacks import (
    register_decision_behaviour_callbacks,
)

from callback_modules.performance_callbacks import (
    register_performance_callbacks,
)

from callback_modules.dataset_callbacks import (
    register_dataset_callbacks,
)


def register_callbacks(app):

    register_dataset_callbacks(app)
    register_feature_importance_callbacks(app)
    register_misclassification_callbacks(app)
    register_decision_behaviour_callbacks(app)
    register_performance_callbacks(app)
