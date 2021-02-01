import abc
import numpy as np
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle
from typing import Tuple


class FakeExperimentOracle(BaseOracle):
    identifier = "E"
    cost = 1

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
