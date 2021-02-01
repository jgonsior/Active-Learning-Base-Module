import abc
import numpy as np
from typing import Tuple
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle
import random


class RandomOracle(BaseOracle):
    identifier = "R"
    cost = -1

    def has_new_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> bool:
        return True

    def get_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
        return query_indices, np.array(
            [
                random.randint(
                    0, len(active_learner.data_storage.label_encoder.classes_)
                )
                for _ in range(0, len(query_indices))
            ]
        )
