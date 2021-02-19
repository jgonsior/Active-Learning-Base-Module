import numpy as np

from active_learning.dataStorage import DataStorage, IndiceMask
from active_learning.learner.standard import Learner
from active_learning.sampling_strategies.BaseSamplingStrategy import \
    BaseSamplingStrategy


class RandomSampler(BaseSamplingStrategy):
    def what_to_label_next(
        self, BATCH_SIZE: int, _, data_storage: DataStorage
    ) -> IndiceMask:
        return np.random.choice(
            data_storage.unlabeled_mask,
            size=BATCH_SIZE,
            replace=False,
        )
