import numpy as np
import collections
import random
import pandas as pd

from ..activeLearner import ActiveLearner
from .baseWeakSupervisionStrategy import BaseWeakSupervisionStrategy


class WeakCert(BaseWeakSupervisionStrategy):
    def get_weak_requests(self, CERTAINTY_THRESHOLD, CERTAINTY_RATIO):
        # calculate certainties for all of X_train_unlabeled
        certainties = self.clf.predict_proba(
            self.data_storage.X_train_unlabeled.to_numpy()
        )

        amount_of_certain_labels = np.count_nonzero(
            np.where(np.max(certainties, 1) > CERTAINTY_THRESHOLD)
        )

        if (
            amount_of_certain_labels
            > len(self.data_storage.X_train_unlabeled) * CERTAINTY_RATIO
        ):

            # for safety reasons I refrain from explaining the following
            certain_indices = [
                j
                for i, j in enumerate(
                    self.data_storage.X_train_unlabeled.index.tolist()
                )
                if np.max(certainties, 1)[i] > CERTAINTY_THRESHOLD
            ]

            certain_X = self.data_storage.X_train_unlabeled.loc[certain_indices]

            recommended_labels = self.clf.predict(certain_X.to_numpy())
            # add indices to recommended_labels, could be maybe useful later on?
            recommended_labels = pd.DataFrame(recommended_labels, index=certain_X.index)

            return certain_X, recommended_labels, certain_indices
        else:
            return None, None, None

        return certain_X, recommended_labels, certain_indices
