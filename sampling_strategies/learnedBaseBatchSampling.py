import math
import os

from numba import jit

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import copy

import numpy as np
from sklearn.metrics import pairwise_distances
from .learnedBaseSampling import LearnedBaseSampling


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
    def _calculate_furthest_metric(self, batch_indices):
        return np.sum(pairwise_distances(self.data_storage.X[batch_indices]))

    def _calculate_furthest_lab_metric(self, batch_indices):
        return np.sum(
            pairwise_distances(
                self.data_storage.X[batch_indices],
                self.data_storage.X[self.data_storage.labeled_mask],
            )
        )

    def _calculate_uncertainty_metric(self, batch_indices):
        Y_proba = self.clf.predict_proba(self.data_storage.X[batch_indices])
        margin = np.partition(-Y_proba, 1, axis=1)
        return np.sum(-np.abs(margin[:, 0] - margin[:, 1]))

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
        elif (
            INITIAL_BATCH_SAMPLING_METHOD == "furthest"
            or INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab"
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
                metric_function = self._calculate_furthest_metric
            elif INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab":
                metric_function = self._calculate_furthest_lab_metric
            elif INITIAL_BATCH_SAMPLING_METHOD == "uncertainty":
                metric_function = self._calculate_uncertainty_metric
            metric_values = [metric_function(a) for a in possible_batches]

            # take n samples based on the sorting metric, the rest randomly
            index_batches = [
                x
                for _, x in sorted(
                    zip(metric_values, possible_batches),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][:SAMPLE_SIZE]
        elif INITIAL_BATCH_SAMPLING_METHOD == "hybrid2":
            possible_batches = [
                np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.nr_queries_per_iteration,
                    replace=False,
                )
                for x in range(0, INITIAL_BATCH_SAMPLING_ARG)
            ]

            furthest_index_batches = [
                x
                for _, x in sorted(
                    zip(
                        [self._calculate_furthest_metric(a) for a in possible_batches],
                        possible_batches,
                    ),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][: math.ceil(SAMPLE_SIZE / 2)]

            uncertainty_index_batches = [
                x
                for _, x in sorted(
                    zip(
                        [
                            self._calculate_uncertainty_metric(a)
                            for a in possible_batches
                        ],
                        possible_batches,
                    ),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][: math.floor(SAMPLE_SIZE / 2)]

            index_batches = furthest_index_batches + uncertainty_index_batches

        else:
            raise ("NON EXISTENT INITIAL_SAMPLING_METHOD")

        return index_batches

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        self.calculate_next_query_indices_pre_hook()
        batch_indices = self.get_X_query_index()

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.X_query_index = batch_indices
        X_state = self.calculate_state(
            batch_indices,
            STATE_ARGSECOND_PROBAS=self.STATE_ARGSECOND_PROBAS,
            STATE_DIFF_PROBAS=self.STATE_DIFF_PROBAS,
            STATE_ARGTHIRD_PROBAS=self.STATE_ARGTHIRD_PROBAS,
            STATE_DISTANCES_LAB=self.STATE_DISTANCES_LAB,
            STATE_DISTANCES_UNLAB=self.STATE_DISTANCES_UNLAB,
            STATE_PREDICTED_CLASS=self.STATE_PREDICTED_CLASS,
        )

        self.calculate_next_query_indices_post_hook(X_state)
        return batch_indices[self.get_sorting(None).argmax()]

    def calculate_state(
        self,
        batch_indices,
        STATE_ARGSECOND_PROBAS,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_PREDICTED_CLASS,
        STATE_UNCERTAINTIES,
        STATE_DISTANCES,
    ):
        state_list = []
        if STATE_UNCERTAINTIES:
            state_list += [self._calculate_uncertainty_metric(a) for a in batch_indices]

        if STATE_DISTANCES:
            state_list += [self._calculate_furthest_metric(a) for a in batch_indices]
        if STATE_DISTANCES_LAB:
            state_list += [
                self._calculate_furthest_lab_metric(a) for a in batch_indices
            ]
        print(state_list)
        #  @todo normalise this here somehow! maybe calculate max distance first? or guess max distance as i normalized everything to 0-1 first!
        return np.array(state_list)
