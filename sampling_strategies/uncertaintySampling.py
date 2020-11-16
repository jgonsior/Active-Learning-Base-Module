import numpy as np
from scipy.stats import entropy

from ..activeLearner import ActiveLearner


class UncertaintySampler(ActiveLearner):
    def set_uncertainty_strategy(self, strategy):
        self.strategy = strategy

    def setClassifierClasses(self, classes):
        self.classifier_classes = classes

    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        # recieve predictions and probabilitys
        # for all possible classifications of CLASSIFIER

        Y_temp_proba = self.clf.predict_proba(
            self.data_storage.X[self.data_storage.unlabeled_mask]
        )

        if self.strategy == "least_confident":
            result = 1 - np.amax(Y_temp_proba, axis=1)
        elif self.strategy == "max_margin":
            margin = np.partition(-Y_temp_proba, 1, axis=1)
            result = -np.abs(margin[:, 0] - margin[:, 1])
        elif self.strategy == "entropy":
            result = np.apply_along_axis(entropy, 1, Y_temp_proba)

        # sort indices_of_cluster by argsort
        argsort = np.argsort(-result)
        query_indices = self.data_storage.unlabeled_mask[argsort]

        # return smallest probabilities
        return query_indices[: self.nr_queries_per_iteration]
