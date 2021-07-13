import numpy as np
from typing import List, TYPE_CHECKING, Callable, Tuple

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner
    from active_learning.dataStorage import (
        DataStorage,
        IndiceMask,
        LabelList,
        FeatureList,
    )

from ..learner.standard import Learner

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

    def get_labels(self, X: "FeatureList", learner: Learner) -> "LabelList":
        (
            Y_pred_lf,
            _,
        ) = self.labeling_function(X)

        return Y_pred_lf
