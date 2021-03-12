from active_learning.query_sampling_strategies.BaseQuerySamplingStrategy import (
    BaseQuerySamplingStrategy,
)
import random
from xml.dom.domreg import well_known_implementations
from active_learning.oracles.BaseOracle import BaseOracle
from typing import List, Tuple
import numpy as np

from active_learning.dataStorage import DataStorage, IndiceMask, LabelList
from active_learning.learner.standard import Learner

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner


class RandomQuerySampler(BaseQuerySamplingStrategy):
    def what_to_label_next(self, active_learner: "ActiveLearner") -> IndiceMask:
        # first select random samples
        random_sample_ids = np.random.choice(
            active_learner.data_storage.unlabeled_mask,
            size=active_learner.BATCH_SIZE,
            replace=False,
        )

        return random_sample_ids
