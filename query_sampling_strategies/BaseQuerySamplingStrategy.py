import abc
from ..oracles.BaseOracle import BaseOracle

from typing import List, Tuple

from active_learning.dataStorage import IndiceMask, LabelList

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class BaseQuerySamplingStrategy:
    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def what_to_label_next(self, active_learner: "ActiveLearner") -> List[IndiceMask]:
        """
        Returns a list of QueryIndices to label next
        Why a list? because each list is passed to an oracle at once
        """
        pass