import abc
import numpy as np
import random
from typing import List, TYPE_CHECKING

from .BaseMergeWeakSupervisionLabelStrategy import BaseMergeWeakSupervisionLabelStrategy

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class RandomLabelMergeStrategy(BaseMergeWeakSupervisionLabelStrategy):
    def merge(self, ws_labels_list: List["LabelList"]) -> "LabelList":
        merged_labels = []

        for i in range(0, len(ws_labels_list)):
            merged_labels.append(random.choice(ws_labels_list[i]))
        return np.array(merged_labels)
