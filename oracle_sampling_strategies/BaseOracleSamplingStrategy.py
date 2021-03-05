import abc
from active_learning.query_sampling_strategies.BaseQuerySamplingStrategy import (
    BaseQuerySamplingStrategy,
)
from ..oracles.BaseOracle import BaseOracle

from typing import Dict, List, Tuple

from active_learning.dataStorage import IndiceMask, LabelList

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class BaseOracleSamplingStrategy:
    @abc.abstractmethod
    def get_oracles(
        self, query_indices_list: List[IndiceMask], active_learner: "ActiveLearner"
    ) -> List[BaseOracle]:
        pass