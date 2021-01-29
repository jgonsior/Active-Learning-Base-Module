import abc
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle
import itertools
from typing import Callable


class LabeleingFunctionsOracle(BaseOracle):
    identifier = "F"

    def __init__(
        self,
        labeling_function: Callable[[Features], Tuple[[QueryIndice], [Label]]],
        cost: float,
    ):
        super().__init__()

        self.labeling_function = labeling_function
        self.cost = cost

    def has_new_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> bool:
        self.potentially_labeled_query_indices, self.Y_lf = self.labeling_function(
            active_learner.data_storage.X[active_learner.data_storage.unlabeled_mask]
        )
        if len(self.potentially_labeled_query_indices) > 0:
            return False
        else:
            return True

    def get_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> tuple[list[QueryIndice], list[Label]]:
        return self.potentially_labeled_query_indices, self.Y_lf
