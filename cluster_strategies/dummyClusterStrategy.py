import numpy as np

from .baseClusterStrategy import BaseClusterStrategy


class DummyClusterStrategy(BaseClusterStrategy):
    def set_data_storage(self, data_storage, n_jobs=-1):
        self.data_storage = data_storage
        self.data_storage.train_unlabeled_cluster_indices = np.ones(
            len(self.data_storage.Y)
        )

    def get_cluster_indices(self, **kwargs):
        return self.data_storage.train_unlabeled_cluster_indices
