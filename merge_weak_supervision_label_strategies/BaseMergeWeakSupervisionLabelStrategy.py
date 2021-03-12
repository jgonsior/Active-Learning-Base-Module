import abc


from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class BaseMergeWeakSupervisionLabelStrategy(abc.ABC):
    @abc.abstractmethod
    def merge(self, ws_labels_list: List["LabelList"]) -> "LabelList":
        pass