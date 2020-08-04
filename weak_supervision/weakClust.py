import collections

from .baseWeakSupervision import BaseWeakSupervision


class WeakClust(BaseWeakSupervision):
    # threshold params
    MINIMUM_CLUSTER_UNITY_SIZE = MINIMUM_RATIO_LABELED_UNLABELED = None

    def get_labeled_samples(self):
        recommended_labels = certain_indices = None
        cluster_found = False

        # check if the most prominent label for one cluster can be propagated over to the rest of it's cluster
        for (
            cluster_id,
            cluster_indices,
        ) in self.data_storage.train_labeled_cluster_indices.items():
            if (
                cluster_id
                not in self.data_storage.train_unlabeled_cluster_indices.keys()
            ):
                continue
            if (
                len(cluster_indices)
                / len(self.data_storage.train_unlabeled_cluster_indices[cluster_id])
                > self.MINIMUM_CLUSTER_UNITY_SIZE
            ):
                frequencies = collections.Counter(
                    self.data_storage.train_labeled_Y.loc[cluster_indices][
                        "label"
                    ].tolist()
                )

                if (
                    frequencies.most_common(1)[0][1]
                    > len(cluster_indices) * self.MINIMUM_RATIO_LABELED_UNLABELED
                ):
                    certain_indices = self.data_storage.train_unlabeled_cluster_indices[
                        cluster_id
                    ]

                    recommended_labels = [
                        frequencies.most_common(1)[0][0] for _ in certain_indices
                    ]
                    #  log_it("Cluster ", cluster_id, certain_indices)
                    cluster_found = True
                    break

        # delete this cluster from the list of possible cluster for the next round
        #  if cluster_found:
        #      self.data_storage.train_labeled_cluster_indices.pop(cluster_id)
        return recommended_labels, certain_indices, "C"
