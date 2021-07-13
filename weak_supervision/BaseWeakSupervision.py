import abc


from typing import Dict, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from active_learning.dataStorage import IndiceMask, LabelList, FeatureList
    from active_learning.dataStorage import DataStorage
from ..learner.standard import Learner


class BaseWeakSupervision(abc.ABC):
    identifier: str

    @abc.abstractmethod
    def get_labels(self, X: "FeatureList", learner: Learner) -> "LabelList":
        pass
