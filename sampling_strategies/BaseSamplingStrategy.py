import abc
from ..oracles.BaseOracle import BaseOracle

from typing import Tuple

from active_learning.dataStorage import IndiceMask, LabelList

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class BaseSamplingStrategy:
    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def what_to_label_next(
        self, active_learner: "ActiveLearner"
    ) -> Tuple[IndiceMask, LabelList, BaseOracle]:
        pass