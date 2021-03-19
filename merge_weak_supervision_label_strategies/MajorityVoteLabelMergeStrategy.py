import abc
from collections import Counter
from .BaseMergeWeakSupervisionLabelStrategy import BaseMergeWeakSupervisionLabelStrategy
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class MajorityVoteLabelMergeStrategy(BaseMergeWeakSupervisionLabelStrategy):
    def merge(self, ws_labels_list: List["LabelList"]) -> "LabelList":
        merged_labels = []
        for i in range(0, len(ws_labels_list[0])):
            c = Counter(ws_labels_list[i])
            merged_labels[i] = c.most_common(1)[0][0]

        return np.array(merged_labels)
