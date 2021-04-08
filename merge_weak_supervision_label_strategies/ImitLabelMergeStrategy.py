import abc
from typing import List, TYPE_CHECKING

from .BaseMergeWeakSupervisionLabelStrategy import BaseMergeWeakSupervisionLabelStrategy

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class ImitLabelMergeStrategy(BaseMergeWeakSupervisionLabelStrategy):
    def merge(self, ws_labels_list: List["LabelList"]) -> "LabelList":
        pass
