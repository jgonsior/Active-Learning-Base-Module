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
from .learnedBaseSampling import LearnedBaseSampling


class LearnedBaseBatchSampling(LearnedBaseSampling):
    def sample_unlabeled_X(
        self,
        sample_size,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
    ):
        index_batches = []
        if INITIAL_BATCH_SAMPLING_METHOD == "random":
            for _ in range(0, sample_size):
                index_batches.append(
                    np.random.choice(
                        self.data_storage.unlabeled_mask,
                        size=self.nr_queries_per_iteration,
                        replace=False,
                    )
                )
        print(index_batches)
        return index_batches
        #  elif INITIAL_BATCH_SAMPLING_METHOD == "furthest":
        #      max_sum = 0
        #      for i in range(0, INITIAL_BATCH_SAMPLING_ARG):
        #          random_index = np.random.choice(
        #              self.data_storage.unlabeled_mask, size=sample_size, replace=False
        #          )
        #          random_sample = self.data_storage.X[random_index]
        #
        #          # calculate distance to each other
        #          total_distance = np.sum(
        #              pairwise_distances(random_sample, random_sample)
        #          )
        #
        #          total_distance += np.sum(
        #              pairwise_distances(
        #                  random_sample,
        #                  self.data_storage.X[self.data_storage.labeled_mask],
        #              )
        #          )
        #          if total_distance > max_sum:
        #              max_sum = total_distance
        #              X_query_index = random_index
        #
        #  elif INITIAL_BATCH_SAMPLING_METHOD == "graph_density":
        #      graph_density = copy.deepcopy(self.data_storage.graph_density)
        #
        #      initial_sample_indexes = []
        #
        #      for _ in range(1, sample_size):
        #          selected = np.argmax(graph_density)
        #          neighbors = (self.data_storage.connect_lil[selected, :] > 0).nonzero()[
        #              1
        #          ]
        #          graph_density[neighbors] = (
        #              graph_density[neighbors] - graph_density[selected]
        #          )
        #          initial_sample_indexes.append(selected)
        #          graph_density[initial_sample_indexes] = min(graph_density) - 1
        #
        #      X_query_index = self.data_storage.initial_unlabeled_mask[
        #          initial_sample_indexes
        #      ]
        #

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        self.calculate_next_query_indices_pre_hook()
        batch_indices = self.get_X_query_index()
        print(self.get_sorting(None))
        print(self.get_sorting(None).argmax())
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
