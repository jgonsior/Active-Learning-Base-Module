import numpy as np

from ..activeLearner import ActiveLearner


class RandomSampler(ActiveLearner):
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices):
        return np.random.choice(
            self.data_storage.unlabeled_mask,
            size=self.NR_QUERIES_PER_ITERATION,
            replace=False,
        )
