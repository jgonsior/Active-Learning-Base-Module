import random
from typing import TYPE_CHECKING, Tuple

import numpy as np
if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner
from .BaseOracle import BaseOracle


class RandomOracle(BaseOracle):
    identifier = "R"
    cost = -1

    def has_new_labels(
        self, query_indices: np.ndarray, active_learner: 'ActiveLearner'
    ) -> bool:
        return True

    def get_labels(
        self, query_indices: np.ndarray, active_learner: 'ActiveLearner'
    ) -> Tuple[np.ndarray, np.ndarray]:
        return query_indices, np.array(
            [
                random.randint(
                    0, len(active_learner.data_storage.label_encoder.classes_) # type: ignore
                )
                for _ in range(0, len(query_indices))
            ]
        )
