import abc
from collections import Counter
from .BaseMergeWeakSupervisionLabelStrategy import BaseMergeWeakSupervisionLabelStrategy
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class MajorityVoteLabelMergeStrategy(BaseMergeWeakSupervisionLabelStrategy):
    def merge(self, ws_labels_array: np.ndarray) -> "LabelList":
        merged_labels = np.ones(ws_labels_array.shape[0]) * -1

        for i in range(0, len(ws_labels_array[0])):
            c = Counter(ws_labels_array[i])
            most_common = c.most_common(1)[0][0]

            if most_common == "-1":
                most_common = c.most_common(1)[1][0]

            merged_labels[i] = most_common
        return merged_labels
