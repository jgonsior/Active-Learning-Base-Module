from active_learning.query_sampling_strategies.BaseQuerySamplingStrategy import (
    BaseQuerySamplingStrategy,
)
import numpy as np
from scipy.stats import entropy

from active_learning.dataStorage import IndiceMask
from active_learning.learner.standard import Learner

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class UncertaintyQuerySampler(BaseQuerySamplingStrategy):
    def __init__(self, strategy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def what_to_label_next(self, active_learner: "ActiveLearner") -> IndiceMask:
        Y_temp_proba = active_learner.learner.predict_proba(
            active_learner.data_storage.X[active_learner.data_storage.unlabeled_mask]
        )

        if self.strategy == "least_confident":
            result = 1 - np.amax(Y_temp_proba, axis=1)
        elif self.strategy == "max_margin":
            margin = np.partition(-Y_temp_proba, 1, axis=1)  # type: ignore
            result = -np.abs(margin[:, 0] - margin[:, 1])
        elif self.strategy == "entropy":
            result = np.apply_along_axis(entropy, 1, Y_temp_proba)  # type: ignore

        # sort indices_of_cluster by argsort
        argsort = np.argsort(-result)  # type: ignore
        query_indices = active_learner.data_storage.unlabeled_mask[argsort]

        # return smallest probabilities
        return query_indices[: active_learner.BATCH_SIZE]
