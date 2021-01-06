import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import copy

import numpy as np
from sklearn.metrics import pairwise_distances
import abc
from ..activeLearner import ActiveLearner


class LearnedBaseSampling(ActiveLearner):
    def sample_unlabeled_X(
        self,
        sample_size,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
    ):
        if INITIAL_BATCH_SAMPLING_METHOD == "random":
            X_query_index = np.random.choice(
                self.data_storage.unlabeled_mask, size=sample_size, replace=False
            )
        elif INITIAL_BATCH_SAMPLING_METHOD == "furthest":
            max_sum = 0
            for i in range(0, INITIAL_BATCH_SAMPLING_ARG):
                random_index = np.random.choice(
                    self.data_storage.unlabeled_mask, size=sample_size, replace=False
                )
                random_sample = self.data_storage.X[random_index]

                # calculate distance to each other
                total_distance = np.sum(
                    pairwise_distances(
                        random_sample, random_sample, metric=self.DISTANCE_METRIC
                    )
                )

                total_distance += np.sum(
                    pairwise_distances(
                        random_sample,
                        self.data_storage.X[self.data_storage.labeled_mask],
                        metric=self.DISTANCE_METRIC,
                    )
                )
                if total_distance > max_sum:
                    max_sum = total_distance
                    X_query_index = random_index

        else:
            print(
                "No valid INITIAL_BATCH_SAMPLING_METHOD given. Valid for single are random or furthest. You specified: "
                + INITIAL_BATCH_SAMPLING_METHOD
            )
            raise ()
        return X_query_index

    @abc.abstractmethod
    def calculate_next_query_indices_pre_hook(self):
        pass

    @abc.abstractmethod
    def get_X_query_index(self):
        pass

    @abc.abstractmethod
    def calculate_next_query_indices_post_hook(self, X_state):
        pass

    @abc.abstractmethod
    def get_sorting(self, X_state):
        pass

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        self.calculate_next_query_indices_pre_hook()
        X_query_index = self.get_X_query_index()

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.X_query_index = X_query_index

        X_state = self.calculate_state(
            self.data_storage.X[X_query_index],
        )

        self.calculate_next_query_indices_post_hook(X_state)

        # use the optimal values
        zero_to_one_values_and_index = list(
            zip(self.get_sorting(X_state), X_query_index)
        )
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.X_query_index = X_query_index

        return [
            v
            for k, v in ordered_list_of_possible_sample_indices[
                : self.NR_QUERIES_PER_ITERATION
            ]
        ]

    def calculate_state(
        self,
        X_query,
    ):
        possible_samples_probas = self.clf.predict_proba(X_query)

        sorted_probas = -np.sort(-possible_samples_probas, axis=1)
        argmax_probas = sorted_probas[:, 0]

        state_list = argmax_probas.tolist()

        if self.STATE_ARGSECOND_PROBAS:
            argsecond_probas = sorted_probas[:, 1]
            state_list += argsecond_probas.tolist()
        if self.STATE_DIFF_PROBAS:
            state_list += (argmax_probas - sorted_probas[:, 1]).tolist()
        if self.STATE_ARGTHIRD_PROBAS:
            if np.shape(sorted_probas)[1] < 3:
                state_list += [0 for _ in range(0, len(X_query))]
            else:
                state_list += sorted_probas[:, 2].tolist()
        if self.STATE_PREDICTED_CLASS:
            state_list += self.clf.predict(X_query).tolist()

        if self.STATE_DISTANCES_LAB:
            # calculate average distance to labeled and average distance to unlabeled samples
            average_distance_labeled = (
                np.sum(
                    pairwise_distances(
                        self.data_storage.X[self.data_storage.labeled_mask],
                        X_query,
                        metric=self.DISTANCE_METRIC,
                    ),
                    axis=0,
                )
                / len(self.data_storage.X[self.data_storage.labeled_mask])
            )
            state_list += average_distance_labeled.tolist()

        if self.STATE_DISTANCES_UNLAB:
            # calculate average distance to labeled and average distance to unlabeled samples
            average_distance_unlabeled = (
                np.sum(
                    pairwise_distances(
                        self.data_storage.X[self.data_storage.unlabeled_mask],
                        X_query,
                        metric=self.DISTANCE_METRIC,
                    ),
                    axis=0,
                )
                / len(self.data_storage.X[self.data_storage.unlabeled_mask])
            )
            state_list += average_distance_unlabeled.tolist()

        if self.STATE_INCLUDE_NR_FEATURES:
            state_list = [self.data_storage.X.shape[1]] + state_list
        return np.array(state_list)
