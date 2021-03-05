from collections import defaultdict
from active_learning.oracle_sampling_strategies.BaseOracleSamplingStrategy import (
    BaseOracleSamplingStrategy,
)
import random
from active_learning.oracles.BaseOracle import BaseOracle
from typing import Dict, List, Tuple
import numpy as np

from active_learning.dataStorage import DataStorage, IndiceMask, LabelList
from active_learning.learner.standard import Learner


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class RandomOracleSamplingStrategy(BaseOracleSamplingStrategy):
    def get_oracles(
        self, query_indices_list: List[IndiceMask], active_learner: "ActiveLearner"
    ) -> List[BaseOracle]:
        # ask each of the oracles where they can provide labels
        potential_oracles: Dict[IndiceMask, List[BaseOracle]] = defaultdict(
            lambda: list()
        )

        for query_indices in query_indices_list:
            for oracle in active_learner.oracles:
                if oracle.has_new_labels(query_indices, active_learner):
                    potential_oracles[query_indices].append(oracle)
                    continue

        oracle_per_queries: List[BaseOracle] = []

        # then select randomly oracles per indices
        for query_indices in query_indices_list:
            oracle_per_queries.append(random.choice(potential_oracles[query_indices]))

        return oracle_per_queries