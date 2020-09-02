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
        REPRESENTATIVE_FEATURES,
        CONVEX_HULL_SAMPLING,
        NO_DIFF_FEATURES,
        LRU_AREAS_LIMIT,
    ):
        self.REPRESENTATIVE_FEATURES = REPRESENTATIVE_FEATURES
        self.CONVEX_HULL_SAMPLING = CONVEX_HULL_SAMPLING
        self.NO_DIFF_FEATURES = NO_DIFF_FEATURES
        self.LRU_AREAS_LIMIT = LRU_AREAS_LIMIT
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
            OLD=self.REPRESENTATIVE_FEATURES,
            LRU_AREAS_LIMIT=self.LRU_AREAS_LIMIT,
            NO_DIFF_FEATURES=self.NO_DIFF_FEATURES,
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

        if self.LRU_AREAS_LIMIT > 0:
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
            if len(self.lru_samples) > self.LRU_AREAS_LIMIT:
                self.lru_samples = self.lru_samples.tail(self.LRU_AREAS_LIMIT)

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
        OLD=False,
        NO_DIFF_FEATURES=False,
        LRU_AREAS_LIMIT=0,
        lru_samples=[],
    ):
        possible_samples_probas = self.clf.predict_proba(X_query)

        sorted_probas = -np.sort(-possible_samples_probas, axis=1)
        argmax_probas = sorted_probas[:, 0]
        argsecond_probas = sorted_probas[:, 1]

        if OLD:
            if NO_DIFF_FEATURES:
                arg_diff_probas = argmax_probas - argsecond_probas
                argsecond_probas = arg_diff_probas
            return np.array([*argmax_probas, *argsecond_probas])

        arg_diff_probas = argmax_probas - argsecond_probas

        if not NO_DIFF_FEATURES:
            arg_diff_probas = argsecond_probas

        if LRU_AREAS_LIMIT == 0:
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

            X_state = np.array(
                [
                    *argmax_probas,
                    *arg_diff_probas,
                    *average_distance_labeled,
                    *average_distance_unlabeled,
                ]
            )
        else:
            if len(lru_samples) == 0:
                lru_distances = [0 for _ in range(0, len(X_query))]
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
                )
            X_state = np.array([*argmax_probas, *arg_diff_probas, *lru_distances])
        return X_state
