import os
from sklearn.metrics import pairwise_distances
import random
from itertools import chain

import dill
import numpy as np

from .imitationLearningSampling import calculate_state, sample_unlabeled_X
from ..activeLearner import ActiveLearner


class TrainedNNLearner(ActiveLearner):
    def init_sampling_classifier(
        self,
        NN_BINARY_PATH,
        AMOUNT_OF_RANDOM_QUERY_SETS,
        REPRESENTATIVE_FEATURES,
        CONVEX_HULL_SAMPLING,
    ):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model
        self.AMOUNT_OF_RANDOM_QUERY_SETS = AMOUNT_OF_RANDOM_QUERY_SETS
        self.REPRESENTATIVE_FEATURES = REPRESENTATIVE_FEATURES
        self.CONVEX_HULL_SAMPLING = CONVEX_HULL_SAMPLING

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        train_unlabeled_X_indices = list(
            chain(*list(train_unlabeled_X_cluster_indices.values()))
        )

        # do this n couple of times to make out of the semi pairwise a real listwise???

        zero_to_one_values_and_index = []
        for _ in range(0, self.AMOUNT_OF_RANDOM_QUERY_SETS):
            X_query = sample_unlabeled_X(
                self.data_storage.train_unlabeled_X,
                self.data_storage.train_labeled_X,
                self.sampling_classifier.n_outputs_,
                self.CONVEX_HULL_SAMPLING,
            )
            possible_samples_indices = X_query.index

            X_state = calculate_state(
                X_query,
                self.data_storage,
                self.clf,
                old=not self.REPRESENTATIVE_FEATURES,
            )
            X_state = np.reshape(X_state, (1, len(X_state)))

            Y_pred = self.sampling_classifier.predict(X_state)

            sorting = Y_pred

            zero_to_one_values_and_index += list(zip(sorting, possible_samples_indices))

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.possible_samples_indices = [
                ix for proba, ix, in zero_to_one_values_and_index
            ]
            #  self.data_storage.possible_samples_indices = zero_to_one_values_and_index

        #  print(zero_to_one_values_and_index)
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return [
            v
            for k, v in ordered_list_of_possible_sample_indices[
                : self.nr_queries_per_iteration
            ]
        ]
