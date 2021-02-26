from active_learning.dataStorage import DataStorage, FeatureList, IndiceMask, LabelList
from typing import List, TYPE_CHECKING, Callable, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner

from .BaseOracle import BaseOracle

LabelConfidence = np.ndarray


class LabeleingFunctionsOracle(BaseOracle):
    identifier = "F"

    def __init__(
        self,
        labeling_function: Callable[
            [DataStorage], Tuple[IndiceMask, LabelList, LabelConfidence]
        ],
        cost: float,
    ):
        super().__init__()

        self.labeling_function = labeling_function
        self.cost = cost

    def has_new_labels(
        self, query_indices: IndiceMask, active_learner: "ActiveLearner"
    ) -> bool:
        (
            self.X_query_indices_lf,
            self.Y_pred_lf,
            self.Y_probas_lf,
        ) = self.labeling_function(active_learner.data_storage)
        if len(self.X_query_indices_lf) > 0:
            return True
        else:
            return False

    def get_labels(
        self, query_indices: IndiceMask, active_learner: "ActiveLearner"
    ) -> Tuple[IndiceMask, LabelList]:
        return self.X_query_indices_lf, self.Y_pred_lf