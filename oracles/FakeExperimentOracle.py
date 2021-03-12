from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.dataStorage import IndiceMask, LabelList
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner
from .BaseOracle import BaseOracle


class FakeExperimentOracle(BaseOracle):
    identifier = "E"

    def get_labels(
        self, query_indices: "IndiceMask", active_learner: "ActiveLearner"
    ) -> Tuple["IndiceMask", "LabelList"]:
        return query_indices, active_learner.data_storage.get_experiment_labels(
            query_indices
        )
