import abc
import numpy as np
from typing import Tuple, List
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle
import itertools
from typing import Callable


class LabeleingFunctionsOracle(BaseOracle):
    identifier = "F"

    def __init__(
        self,
        labeling_function: Callable[
            [np.float64], Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]
        ],
        cost: float,
    ):
        super().__init__()

        self.labeling_function = labeling_function
        self.cost = cost

    def has_new_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> bool:
        self.potentially_labeled_query_indices, self.Y_lf = self.labeling_function(
            active_learner.data_storage.X[active_learner.data_storage.unlabeled_mask]
        )
        if len(self.potentially_labeled_query_indices) > 0:
            return False
        else:
            return True

    def get_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
        return self.potentially_labeled_query_indices, self.Y_lf
