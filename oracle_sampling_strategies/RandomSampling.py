import random
from xml.dom.domreg import well_known_implementations
from active_learning.oracles.BaseOracle import BaseOracle
from typing import List, Tuple
import numpy as np

from active_learning.dataStorage import DataStorage, IndiceMask, LabelList
from active_learning.learner.standard import Learner
from active_learning.sampling_strategies.BaseSamplingStrategy import (
    BaseSamplingStrategy,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class RandomSampler(BaseSamplingStrategy):
    def what_to_label_next(
        self, active_learner: "ActiveLearner"
    ) -> Tuple[IndiceMask, LabelList, BaseOracle]:
        # first select random samples
        random_sample_ids = np.random.choice(
            active_learner.data_storage.unlabeled_mask,
            size=active_learner.BATCH_SIZE,
            replace=False,
        )

        # ask each of the oracles if they can provide a label
        potential_oracles: List[BaseOracle] = []
        for oracle in active_learner.oracles:
            if oracle.has_new_labels(random_sample_ids, active_learner):
                potential_oracles.append(oracle)

        # then select randomly oracle
        random_oracle = random.choice(potential_oracles)
        query_indices, new_labels = random_oracle.get_labels(
            random_sample_ids, active_learner
        )

        return (
            query_indices,
            new_labels,
            random_oracle,
        )
