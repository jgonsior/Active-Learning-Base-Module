import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import statistics
import copy
import random
from itertools import chain

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import abc
from ..activeLearner import ActiveLearner


class LearnedBaseSampling(ActiveLearner):
    def init_sampling_classifier(
        self,
        CONVEX_HULL_SAMPLING,
        STATE_DISTANCES,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_LRU_AREAS_LIMIT,
        STATE_PREDICTED_CLASS,
        STATE_ARGSECOND_PROBAS,
        STATE_NO_LRU_WEIGHTS,
    ):
        self.CONVEX_HULL_SAMPLING = CONVEX_HULL_SAMPLING
        self.STATE_DISTANCES = STATE_DISTANCES
        self.STATE_DISTANCES_LAB = STATE_DISTANCES_LAB
        self.STATE_DISTANCES_UNLAB = STATE_DISTANCES_UNLAB
        self.STATE_DIFF_PROBAS = STATE_DIFF_PROBAS
        self.STATE_ARGTHIRD_PROBAS = STATE_ARGTHIRD_PROBAS
        self.STATE_LRU_AREAS_LIMIT = STATE_LRU_AREAS_LIMIT
        self.STATE_ARGSECOND_PROBAS = STATE_ARGSECOND_PROBAS
        self.STATE_NO_LRU_WEIGHTS = STATE_NO_LRU_WEIGHTS
        self.STATE_PREDICTED_CLASS = STATE_PREDICTED_CLASS

        self.lru_samples = pd.DataFrame(
            data=None, columns=self.data_storage.train_unlabeled_X.columns, index=None
        )

    @abc.abstractmethod
    def calculate_next_query_indices_pre_hook(self):
        pass

    @abc.abstractmethod
    def get_X_query(self):
        pass

    @abc.abstractmethod
    def calculate_next_query_indices_post_hook(self, X_state):
        pass

    @abc.abstractmethod
    def get_sorting(self, X_state):
        pass

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        self.calculate_next_query_indices_pre_hook()
        X_query = self.get_X_query()
        possible_samples_indices = X_query.index

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.possible_samples_indices = X_query.index

        X_state = self.calculate_state(
            X_query,
            STATE_ARGSECOND_PROBAS=self.STATE_ARGSECOND_PROBAS,
            STATE_DIFF_PROBAS=self.STATE_DIFF_PROBAS,
            STATE_ARGTHIRD_PROBAS=self.STATE_ARGTHIRD_PROBAS,
            STATE_LRU_AREAS_LIMIT=self.STATE_LRU_AREAS_LIMIT,
            STATE_DISTANCES=self.STATE_DISTANCES,
            STATE_DISTANCES_LAB=self.STATE_DISTANCES_LAB,
            STATE_DISTANCES_UNLAB=self.STATE_DISTANCES_UNLAB,
            STATE_PREDICTED_CLASS=self.STATE_PREDICTED_CLASS,
            STATE_NO_LRU_WEIGHTS=self.STATE_NO_LRU_WEIGHTS,
            lru_samples=self.lru_samples,
        )

        self.calculate_next_query_indices_post_hook(X_state)

        # use the optimal values
        zero_to_one_values_and_index = list(
            zip(self.get_sorting(X_state), possible_samples_indices)
        )
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.possible_samples_indices = possible_samples_indices

        if self.STATE_LRU_AREAS_LIMIT > 0:
            # fifo queue
            self.lru_samples = pd.concat(
                [
                    self.lru_samples,
                    self.data_storage.train_unlabeled_X.loc[
                        [
                            v
                            for k, v in ordered_list_of_possible_sample_indices[
                                : self.nr_queries_per_iteration
                            ]
                        ]
                    ],
                ],
                #  ignore_index=True,
            )

            # clean up
            if len(self.lru_samples) > self.STATE_LRU_AREAS_LIMIT:
                self.lru_samples = self.lru_samples.tail(self.STATE_LRU_AREAS_LIMIT)

        return [
            v
            for k, v in ordered_list_of_possible_sample_indices[
                : self.nr_queries_per_iteration
            ]
        ]

    def sample_unlabeled_X(
        self, train_unlabeled_X, train_labeled_X, sample_size, CONVEX_HULL_SAMPLING
    ):
        if CONVEX_HULL_SAMPLING:
            max_sum = 0
            for i in range(0, 100):
                random_sample = train_unlabeled_X.sample(n=sample_size)

                # calculate distance to each other
                total_distance = np.sum(
                    pairwise_distances(random_sample, random_sample)
                )
                #  total_distance += np.sum(
                #      pairwise_distances(random_sample, train_unlabeled_X)
                #  )
                total_distance += np.sum(
                    pairwise_distances(random_sample, train_labeled_X)
                )
                if total_distance > max_sum:
                    max_sum = total_distance
                    X_query = random_sample
            possible_samples_indices = X_query.index

        else:
            X_query = train_unlabeled_X.sample(n=sample_size)
        return X_query

    def calculate_state(
        self,
        X_query,
        STATE_ARGSECOND_PROBAS,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_LRU_AREAS_LIMIT,
        STATE_DISTANCES,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_PREDICTED_CLASS,
        STATE_NO_LRU_WEIGHTS,
        lru_samples=[],
    ):
        possible_samples_probas = self.clf.predict_proba(X_query)

        sorted_probas = -np.sort(-possible_samples_probas, axis=1)
        argmax_probas = sorted_probas[:, 0]

        state_list = argmax_probas.tolist()

        if STATE_ARGSECOND_PROBAS:
            argsecond_probas = sorted_probas[:, 1]
            state_list += argsecond_probas.tolist()
        if STATE_DIFF_PROBAS:
            state_list += (argmax_probas - sorted_probas[:, 1]).tolist()
        if STATE_ARGTHIRD_PROBAS:
            if np.shape(sorted_probas)[1] < 3:
                state_list += [0 for _ in range(0, len(X_query))]
            else:
                state_list += sorted_probas[:, 2].tolist()
        if STATE_PREDICTED_CLASS:
            state_list += self.clf.predict(X_query).tolist()

        if STATE_DISTANCES_LAB:
            # calculate average distance to labeled and average distance to unlabeled samples
            average_distance_labeled = (
                np.sum(
                    pairwise_distances(self.data_storage.train_labeled_X, X_query),
                    axis=0,
                )
                / len(self.data_storage.train_labeled_X)
            )
            state_list += average_distance_labeled.tolist()

        if STATE_DISTANCES_UNLAB:
            # calculate average distance to labeled and average distance to unlabeled samples
            average_distance_unlabeled = (
                np.sum(
                    pairwise_distances(self.data_storage.train_unlabeled_X, X_query),
                    axis=0,
                )
                / len(self.data_storage.train_unlabeled_X)
            )
            state_list += average_distance_unlabeled.tolist()

        if STATE_DISTANCES:
            # calculate average distance to labeled and average distance to unlabeled samples
            average_distance_labeled = (
                np.sum(
                    pairwise_distances(self.data_storage.train_labeled_X, X_query),
                    axis=0,
                )
                / len(self.data_storage.train_labeled_X)
            )
            average_distance_unlabeled = (
                np.sum(
                    pairwise_distances(self.data_storage.train_unlabeled_X, X_query),
                    axis=0,
                )
                / len(self.data_storage.train_unlabeled_X)
            )
            state_list += average_distance_labeled.tolist()
            state_list += average_distance_unlabeled.tolist()

        if STATE_LRU_AREAS_LIMIT > 0:
            if len(lru_samples) == 0:
                lru_distances = [len(lru_samples) + 2 for _ in range(0, len(X_query))]
            else:
                if STATE_NO_LRU_WEIGHTS:
                    lru_distances = (
                        np.sum(pairwise_distances(X_query, lru_samples), axis=1)
                        / len(lru_samples)
                    ).tolist()
                else:
                    lru_distances = (
                        np.sum(
                            np.multiply(
                                [i for i in range(1, len(lru_samples) + 1)],
                                pairwise_distances(X_query, lru_samples),
                            ),
                            axis=1,
                        )
                        / len(lru_samples)
                    ).tolist()
            state_list += lru_distances

        return np.array(state_list)
