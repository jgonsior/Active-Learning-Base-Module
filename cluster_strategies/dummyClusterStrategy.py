from .baseClusterStrategy import BaseClusterStrategy


class DummyClusterStrategy(BaseClusterStrategy):
    def get_cluster_indices(self, **kwargs):
        return self.data_storage.train_unlabeled_cluster_indices
