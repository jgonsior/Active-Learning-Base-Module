import abc
from .BaseMergeWeakSupervisionLabelStrategy import BaseMergeWeakSupervisionLabelStrategy
import numpy as np

from typing import List, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class RandomLabelMergeStrategy(BaseMergeWeakSupervisionLabelStrategy):
    def merge(self, ws_labels_list: List["LabelList"]) -> "LabelList":
        merged_labels = []
        for i in range(0, len(ws_labels_list[0])):
            merged_labels[i] = random.choice(ws_labels_list)[i]
        return np.array(merged_labels)