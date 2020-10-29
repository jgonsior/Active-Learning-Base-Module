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
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import lil_matrix


class LearnedBaseSampling(ActiveLearner):
    def init_sampling_classifier(
        self,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_PREDICTED_CLASS,
        STATE_ARGSECOND_PROBAS,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
    ):
        self.STATE_DISTANCES_LAB = STATE_DISTANCES_LAB
        self.STATE_DISTANCES_UNLAB = STATE_DISTANCES_UNLAB
        self.STATE_DIFF_PROBAS = STATE_DIFF_PROBAS
        self.STATE_ARGTHIRD_PROBAS = STATE_ARGTHIRD_PROBAS
        self.STATE_ARGSECOND_PROBAS = STATE_ARGSECOND_PROBAS
        self.STATE_PREDICTED_CLASS = STATE_PREDICTED_CLASS
        self.INITIAL_BATCH_SAMPLING_METHOD = INITIAL_BATCH_SAMPLING_METHOD
        self.INITIAL_BATCH_SAMPLING_ARG = INITIAL_BATCH_SAMPLING_ARG

        self.lru_samples = pd.DataFrame(
            data=None, columns=self.data_storage.train_unlabeled_X.columns, index=None
        )

        if INITIAL_BATCH_SAMPLING_METHOD == "graph_density":
            # compute k-nearest neighbors grap
            self.compute_graph_density()

    # adapted from https://github.com/google/active-learning/blob/master/sampling_methods/graph_density.py#L47-L72
    # original idea: https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf
    def compute_graph_density(self, n_neighbor=10):
        X_df = self.data_storage.get_all_train_X()
        gamma = 1.0 / X_df.shape[1]
        X = X_df.to_numpy()
        # kneighbors graph is constructed using k=10
        connect = kneighbors_graph(
            X, n_neighbor, metric="manhattan"
        )  # , n_jobs=self.N_JOBS)

        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()

        inds = zip(neighbors[0], neighbors[1])

        # changes as in connect[i, j] = new_weight are much faster for lil_matrix
        connect_lil = lil_matrix(connect)

        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]

            # das hier auf einmal berechnen?!
            distance = pairwise_distances(
                X[[i]], X[[j]], metric="manhattan"  # , n_jobs=self.N_JOBS
            )

            distance = distance[0, 0]

            # gaussian kernel
            weight = np.exp(-distance * gamma)
            connect_lil[i, j] = weight
            connect_lil[j, i] = weight

        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.

        self.data_storage.graph_density_labeled = pd.DataFrame(columns=["density"])
        self.data_storage.graph_density_unlabeled = pd.DataFrame(columns=["density"])

        for i in range(0, len(self.data_storage.train_unlabeled_X)):
            self.data_storage.graph_density_unlabeled.loc[X_df.index[i]] = [
                connect_lil[i, :].sum() / (connect_lil[i, :] > 0).sum()
            ]

        for i in range(0, len(self.data_storage.train_labeled_X)):
            self.data_storage.graph_density_labeled.loc[X_df.index[i]] = [
                connect_lil[i, :].sum() / (connect_lil[i, :] > 0).sum()
            ]

        #  self.graph_density = np.zeros(X.shape[0])
        #  for i in np.arange(X.shape[0]):
        #      self.graph_density[i] = (
        #          connect_lil[i, :].sum() / (connect_lil[i, :] > 0).sum()
        #      )
        #  self.starting_density = copy.deepcopy(self.graph_density)
        self.connect = connect_lil

    def sample_unlabeled_X(
        self,
        train_unlabeled_X,
        train_labeled_X,
        sample_size,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
    ):
        if INITIAL_BATCH_SAMPLING_METHOD == "random":
            X_query = train_unlabeled_X.sample(n=sample_size)
        elif INITIAL_BATCH_SAMPLING_METHOD == "furthest":
            max_sum = 0
            for i in range(0, INITIAL_BATCH_SAMPLING_ARG):
                random_sample = train_unlabeled_X.sample(n=sample_size)

                # calculate distance to each other
                total_distance = np.sum(
                    pairwise_distances(random_sample, random_sample)
                )

                total_distance += np.sum(
                    pairwise_distances(random_sample, train_labeled_X)
                )
                if total_distance > max_sum:
                    max_sum = total_distance
                    X_query = random_sample

        elif INITIAL_BATCH_SAMPLING_METHOD == "graph_density":
            # get n samples with highest connectivity
            print(self.data_storage.graph_density_unlabeled)
            print(self.data_storage.graph_density_unlabeled["density"].to_numpy())
            max_n_local_indices = np.argpartition(
                self.data_storage.graph_density_unlabeled["density"].to_numpy(),
                -sample_size,
            )[-sample_size:]
            print(max_n_local_indices)
            print(self.data_storage.graph_density_unlabeled[max_n_local_indices])
            possible_samples_indices = self.data_storage.graph_density_unlabeled[
                "density"
            ][max_n_local_indices]
            X_query = self.data_storage.train_unlabeled_X[possible_samples_indices]
            exit(-1)
        return X_query

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
            STATE_DISTANCES_LAB=self.STATE_DISTANCES_LAB,
            STATE_DISTANCES_UNLAB=self.STATE_DISTANCES_UNLAB,
            STATE_PREDICTED_CLASS=self.STATE_PREDICTED_CLASS,
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

        return np.array(state_list)
