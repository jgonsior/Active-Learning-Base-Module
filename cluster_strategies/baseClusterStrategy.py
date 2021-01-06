import abc
from collections import Counter, defaultdict
from math import e, log

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class BaseClusterStrategy:
    def _entropy(self, labels):
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0

        # compute entropy
        base = e
        for i in probs:
            ent -= i * log(i, base)
        return ent

    def set_data_storage(self, data_storage, n_jobs=-1):
        self.data_storage = data_storage

        n_samples, n_features = np.shape(self.data_storage.X)

        # then cluster it
        self.cluster_model = AgglomerativeClustering(n_clusters=int(n_samples / 8))
        #  distance_threshold=1,
        #  n_clusters=None,
        #  )
        #  self.plot_cluster()
        #  self.plot_dendrogram()

        #  self.cluster_model = MiniBatchKMeans(
        #  n_clusters=int(n_samples / 5),
        #  batch_size=min(int(n_samples / 100), int(n_features)),
        #  )

        #  self.data_storage.train_unlabeled_data = self.data_storage.get_df().assign(cluster=np.nan)
        cluster = self.cluster_model.fit_predict(self.data_storage.X)

        #  self.cluster_model = OPTICS(min_cluster_size=20, n_jobs=n_jobs)
        #  with np.errstate(divide="ignore"):
        #  self.cluster_model.fit(self.data_storage.X_train_unlabeled)

        #  # fit cluster
        #  self.Y_train_unlabeled_cluster = self.cluster_model.labels_[
        #  self.cluster_model.ordering_
        #  ]
        counter = Counter(cluster)
        self.n_clusters = len([1 for _ in counter.most_common()])

        #  log_it(
        #      "Clustering into "
        #      + str(self.n_clusters)
        #      + " :  "
        #      + str(counter.most_common())
        #  )

        # could be broken
        self.data_storage.train_unlabeled_cluster_indices = defaultdict(lambda: list())
        self.data_storage.train_labeled_cluster_indices = defaultdict(lambda: list())

        for cluster_index, X_train_index in zip(
            cluster, self.data_storage.unlabeled_mask,
        ):
            self.data_storage.train_unlabeled_cluster_indices[cluster_index].append(
                X_train_index
            )

    @abc.abstractmethod
    def get_cluster_indices(self, **kwargs):
        # return X_train_unlabeled
        pass
