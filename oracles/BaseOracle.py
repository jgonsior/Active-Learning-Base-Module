from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.dataStorage import IndiceMask, LabelList
    from ..activeLearner import ActiveLearner
from typing import List, TYPE_CHECKING, Tuple


class BaseOracle(ABC):
    identifier: str

    @abstractmethod
    def get_labels(
        self, query_indices: "IndiceMask", active_learner: "ActiveLearner"
    ) -> "LabelList":
        pass

    def get_oracle_identifier(self) -> str:
        return self.identifier
