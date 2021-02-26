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
            [IndiceMask, DataStorage], Tuple[LabelList, LabelConfidence]
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
            self.Y_pred_lf,
            self.Y_probas_lf,
        ) = self.labeling_function(query_indices, active_learner.data_storage)
        if np.count_nonzero(self.Y_pred_lf) > 0:  # type: ignore
            return True
        else:
            return False

    def get_labels(self, query_indices: IndiceMask, _) -> Tuple[IndiceMask, LabelList]:

        # return only the asked samples
        Y_pred_non_abstain = self.Y_pred_lf[self.Y_pred_lf != -1]
        indices = query_indices[self.Y_pred_lf != -1]
        return np.array(indices), Y_pred_non_abstain