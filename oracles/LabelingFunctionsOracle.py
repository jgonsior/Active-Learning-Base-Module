import itertools
from typing import Callable, TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner
from .BaseOracle import BaseOracle


class LabeleingFunctionsOracle(BaseOracle):
    identifier = "F"

    def __init__(
        self,
        labeling_function: Callable[[np.float64], Tuple[np.ndarray, np.ndarray]],
        cost: float,
    ):
        super().__init__()

        self.labeling_function = labeling_function
        self.cost = cost

    def has_new_labels(
        self, query_indices: np.ndarray, active_learner: 'ActiveLearner'
    ) -> bool:
        self.potentially_labeled_query_indices, self.Y_lf = self.labeling_function(
            active_learner.data_storage.X[active_learner.data_storage.unlabeled_mask]
        )
        if len(self.potentially_labeled_query_indices) > 0:
            return False
        else:
            return True

    def get_labels(
        self, query_indices: np.ndarray, active_learner: 'ActiveLearner'
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.potentially_labeled_query_indices, self.Y_lf
