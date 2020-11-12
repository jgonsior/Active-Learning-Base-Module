import os
from numba import jit

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
from .learnedBaseSampling import LearnedBaseSampling
import math


@jit(nopython=True)
def _find_firsts(items, vec):
    """return the index of the first occurence of item in vec"""
    result = []
    for item in items:
        for i in range(len(vec)):
            if item == vec[i]:
                result.append(i)
                break
    return result


class LearnedBaseBatchSampling(LearnedBaseSampling):
    def calculate_max_margin_uncertainty(self, a):
        Y_proba = self.clf.predict_proba(self.data_storage.X[a])
        margin = np.partition(-Y_proba, 1, axis=1)
        return -np.abs(margin[:, 0] - margin[:, 1])

    def sample_unlabeled_X(
        self,
        SAMPLE_SIZE,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
    ):
        index_batches = []
        if INITIAL_BATCH_SAMPLING_METHOD == "random":
            for _ in range(0, SAMPLE_SIZE):
                index_batches.append(
                    np.random.choice(
                        self.data_storage.unlabeled_mask,
                        size=self.nr_queries_per_iteration,
                        replace=False,
                    )
                )
        elif INITIAL_BATCH_SAMPLING_METHOD == "graph_density":
            graph_density = copy.deepcopy(self.data_storage.graph_density)
            for _ in range(0, SAMPLE_SIZE):

                initial_sample_indexes = []

                for _ in range(1, SAMPLE_SIZE):
                    selected = np.argmax(graph_density)
                    neighbors = (
                        self.data_storage.connect_lil[selected, :] > 0
                    ).nonzero()[1]
                    graph_density[neighbors] = (
                        graph_density[neighbors] - graph_density[selected]
                    )
                    initial_sample_indexes.append(selected)
                    graph_density[initial_sample_indexes] = min(graph_density) - 1

                index_batches.append(
                    self.data_storage.initial_unlabeled_mask[initial_sample_indexes]
                )

        elif (
            INITIAL_BATCH_SAMPLING_METHOD == "furthest"
            or INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab"
            or INITIAL_BATCH_SAMPLING_METHOD == "graph_density2"
            or INITIAL_BATCH_SAMPLING_METHOD == "uncertainty"
        ):
            possible_batches = [
                np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.nr_queries_per_iteration,
                    replace=False,
                )
                for x in range(0, INITIAL_BATCH_SAMPLING_ARG)
            ]

            if INITIAL_BATCH_SAMPLING_METHOD == "furthest":
                metric_values = [
                    np.sum(pairwise_distances(self.data_storage.X[a]))
                    for a in possible_batches
                ]
            elif INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab":
                metric_values = [
                    np.sum(
                        pairwise_distances(
                            self.data_storage.X[a],
                            self.data_storage.X[self.data_storage.labeled_mask],
                        )
                    )
                    for a in possible_batches
                ]
            elif INITIAL_BATCH_SAMPLING_METHOD == "graph_density2":
                metric_values = [
                    np.sum(
                        self.data_storage.graph_density[
                            _find_firsts(a, self.data_storage.initial_unlabeled_mask)
                        ]
                    )
                    for a in possible_batches
                ]
            elif INITIAL_BATCH_SAMPLING_METHOD == "uncertainty":
                metric_values = [
                    np.sum(self.calculate_max_margin_uncertainty(a))
                    for a in possible_batches
                ]
            index_batches = [
                x
                for _, x in sorted(
                    zip(metric_values, possible_batches),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][:SAMPLE_SIZE]
        #  elif INITIAL_BATCH_SAMPLING_METHOD == "UNCERTAINTY":
        # randomly select 5 times as many samples as needed
        # select those with the highest average uncertainty
        else:
            raise ("NON EXISTENT INITIAL_SAMPLING_METHOD")

        return index_batches

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        self.calculate_next_query_indices_pre_hook()
        batch_indices = self.get_X_query_index()
        return batch_indices[self.get_sorting(None).argmax()]

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.X_query_index = batch_indices

        X_state = self.calculate_state(
            self.data_storage.X[X_query_index],
            STATE_ARGSECOND_PROBAS=self.STATE_ARGSECOND_PROBAS,
            STATE_DIFF_PROBAS=self.STATE_DIFF_PROBAS,
            STATE_ARGTHIRD_PROBAS=self.STATE_ARGTHIRD_PROBAS,
            STATE_DISTANCES_LAB=self.STATE_DISTANCES_LAB,
            STATE_DISTANCES_UNLAB=self.STATE_DISTANCES_UNLAB,
            STATE_PREDICTED_CLASS=self.STATE_PREDICTED_CLASS,
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
                : self.nr_queries_per_iteration
            ]
        ]

    def calculate_state(
        self,
        X_query,
        STATE_ARGSECOND_PROBAS,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_PREDICTED_CLASS,
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
                    pairwise_distances(
                        self.data_storage.X[self.data_storage.labeled_mask], X_query
                    ),
                    axis=0,
                )
                / len(self.data_storage.X[self.data_storage.labeled_mask])
            )
            state_list += average_distance_labeled.tolist()

        if STATE_DISTANCES_UNLAB:
            # calculate average distance to labeled and average distance to unlabeled samples
            average_distance_unlabeled = (
                np.sum(
                    pairwise_distances(
                        self.data_storage.X[self.data_storage.unlabeled_mask], X_query
                    ),
                    axis=0,
                )
                / len(self.data_storage.X[self.data_storage.unlabeled_mask])
            )
            state_list += average_distance_unlabeled.tolist()

        return np.array(state_list)
