import numpy as np
from scipy.stats import entropy

from active_learning.dataStorage import DataStorage, IndiceMask
from active_learning.learner.standard import Learner
from active_learning.sampling_strategies.BaseSamplingStrategy import (
    BaseSamplingStrategy,
)

from ..activeLearner import ActiveLearner


class UncertaintySampler(BaseSamplingStrategy):
    def __init__(self, strategy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def what_to_label_next(
        self, NR_QUERIES_PER_ITERATION: int, learner: Learner, data_storage: DataStorage
    ) -> IndiceMask:
        Y_temp_proba = learner.predict_proba(
            data_storage.X[data_storage.unlabeled_mask]
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
        query_indices = data_storage.unlabeled_mask[argsort]

        # return smallest probabilities
        return query_indices[:NR_QUERIES_PER_ITERATION]
