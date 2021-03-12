import abc
from active_learning.combined_query_and_oracle_sampling_strategies.StandardCombinedQueryAndOracleSamplingStrategy import (
    StandardCombinedQueryAndOracleSamplingStrategy,
)
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


class ImitCombinedQueryAndOracleSamplingStrategy(
    StandardCombinedQueryAndOracleSamplingStrategy
):
    pass