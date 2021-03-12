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
    cost = 1

    def has_new_labels(
        self, query_indices: "IndiceMask", active_learner: "ActiveLearner"
    ) -> bool:
        if active_learner.stopping_criteria.stop_is_reached():
            return False
        else:
            return True

    def get_labels(
        self, query_indices: "IndiceMask", active_learner: "ActiveLearner"
    ) -> Tuple["IndiceMask", "LabelList"]:
        return query_indices, active_learner.data_storage.get_experiment_labels(
            query_indices
        )
