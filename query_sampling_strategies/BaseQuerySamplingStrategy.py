import abc
from typing import TYPE_CHECKING

from active_learning.dataStorage import IndiceMask

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class BaseQuerySamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def what_to_label_next(self, active_learner: "ActiveLearner") -> IndiceMask:
        pass
