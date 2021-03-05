import abc
from active_learning.oracle_sampling_strategies.BaseOracleSamplingStrategy import (
    BaseOracleSamplingStrategy,
)
from active_learning.query_sampling_strategies.BaseQuerySamplingStrategy import (
    BaseQuerySamplingStrategy,
)
from ..oracles.BaseOracle import BaseOracle

from typing import Dict, List, Tuple

from active_learning.dataStorage import IndiceMask, LabelList

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class StandardMergedSamplingStrategy:
    query_sampling_strategy: BaseQuerySamplingStrategy
    oracle_sampling_strategy: BaseOracleSamplingStrategy

    def __init__(
        self,
        query_sampling_strategy: BaseQuerySamplingStrategy,
        oracle_sampling_strategy: BaseOracleSamplingStrategy,
    ):
        self.query_sampling_strategy = query_sampling_strategy
        self.oracle_sampling_strategy = oracle_sampling_strategy

    @abc.abstractmethod
    def get_next_query_and_oracles(
        self, active_learner: "ActiveLearner"
    ) -> Tuple[List[IndiceMask], List[BaseOracle]]:
        query_indices: List[
            IndiceMask
        ] = self.query_sampling_strategy.what_to_label_next(active_learner)

        oracles: List[BaseOracle] = self.oracle_sampling_strategy.get_oracles(
            query_indices, active_learner
        )

        return query_indices, oracles