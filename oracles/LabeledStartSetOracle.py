import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner

from .BaseOracle import BaseOracle


class LabeledStartSetOracle(BaseOracle):
    identifier = "S"

    def get_labels(
        self, query_indices: np.ndarray, active_learner: "ActiveLearner"
    ) -> Tuple[np.ndarray, np.ndarray]:
        return query_indices, active_learner.data_storage.get_experiment_labels(
            query_indices
        )
