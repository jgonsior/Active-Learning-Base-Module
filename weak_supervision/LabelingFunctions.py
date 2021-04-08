import numpy as np
from typing import List, TYPE_CHECKING, Callable, Tuple

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner
    from active_learning.dataStorage import (
        DataStorage,
        IndiceMask,
        LabelList,
    )


from .BaseWeakSupervision import BaseWeakSupervision

LabelConfidence = np.ndarray


class LabelingFunctions(BaseWeakSupervision):
    def __init__(
        self,
        labeling_function: Callable[
            ["IndiceMask", "DataStorage"], Tuple["LabelList", LabelConfidence]
        ],
        cost: float,
    ):
        super().__init__()

        self.labeling_function = labeling_function
        self.cost = cost

    def get_labels(
        self, query_indices: "IndiceMask", data_storage: "DataStorage"
    ) -> "LabelList":
        (
            Y_pred_lf,
            _,
        ) = self.labeling_function(query_indices, data_storage)

        return Y_pred_lf
