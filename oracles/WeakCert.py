import abc
import numpy as np
from typing import Tuple
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle
import random
import numpy as np


class WeakCert(BaseOracle):
    identifier = "U"
    cost = 0

    def __init__(self, CERTAINTY_THRESHOLD, CERTAINTY_RATIO):
        super().__init__()

        self.CERTAINTY_THRESHOLD = CERTAINTY_THRESHOLD
        self.CERTAINTY_RATIO = CERTAINTY_RATIO

    def has_new_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> bool:
        # calculate certainties for all of X_train_unlabeled
        self.certainties = active_learner.learner.predict_proba(
            active_learner.data_storage.X[active_learner.data_storage.unlabeled_mask]
        )

        # type: ignore
        amount_of_certain_labels = np.count_nonzero(
            np.where(np.max(self.certainties, 1) > self.CERTAINTY_THRESHOLD)
        )

        if (
            amount_of_certain_labels
            > len(active_learner.data_storage.unlabeled_mask) * self.CERTAINTY_RATIO
        ):
            return True
        else:
            return False

    def get_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
        # for safety reasons I refrain from explaining the following
        certain_indices: np.ndarray[np.int64] = np.array(
            [
                j
                for i, j in enumerate(active_learner.data_storage.unlabeled_mask)
                if np.max(self.certainties, 1)[i] > self.CERTAINTY_THRESHOLD
            ]
        )

        certain_X: np.ndarray[np.float64] = active_learner.data_storage.X[
            certain_indices
        ]

        recommended_labels = active_learner.learner.predict(certain_X)

        return certain_indices, recommended_labels
