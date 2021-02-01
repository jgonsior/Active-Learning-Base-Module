import abc
import numpy as np
from typing import Tuple
from ..activeLearner.activeLearner import ActiveLearner
from .BaseOracle import BaseOracle


class LabeledStartSetOracle(BaseOracle):
    identifier = "S"
    cost = 0

    def has_new_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> bool:
        if active_learner.stopping_criteria.stop_is_reached():
            return False
        else:
            return True

    def get_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
        return query_indices, active_learner.data_storage.get_experiment_labels(
            query_indices
        )
