import os
import random
from itertools import chain

import dill
import numpy as np

from ..activeLearner import ActiveLearner
from .imitationLearningSampling import calculate_state


class TrainedNNLearner(ActiveLearner):
    def init_sampling_classifier(self, NN_BINARY_PATH, AMOUNT_OF_RANDOM_QUERY_SETS):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model
        self.AMOUNT_OF_RANDOM_QUERY_SETS = AMOUNT_OF_RANDOM_QUERY_SETS

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        train_unlabeled_X_indices = list(
            chain(*list(train_unlabeled_X_cluster_indices.values()))
        )

        # do this n couple of times to make out of the semi pairwise a real listwise???

        zero_to_one_values_and_index = []
        for _ in range(0, self.AMOUNT_OF_RANDOM_QUERY_SETS):
            random.shuffle(train_unlabeled_X_indices)
            possible_samples_indices = train_unlabeled_X_indices[
                : self.sampling_classifier.n_outputs_
            ]

            X_state = calculate_state(
                self.data_storage.train_unlabeled_X.loc[possible_samples_indices],
                self.data_storage,
                self.clf,
                old=True,
            )
            X_state = np.reshape(X_state, (1, len(X_state)))

            Y_pred = self.sampling_classifier.predict(X_state)

            sorting = Y_pred

            zero_to_one_values_and_index += list(zip(sorting, possible_samples_indices))

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
