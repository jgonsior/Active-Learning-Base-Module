import random
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

    def _calculate_predicted_unity(self, unlabeled_sample_indices):
        Y_pred = self.clf.predict(self.data_storage.X[unlabeled_sample_indices])
        Y_pred_sorted = sorted(Y_pred)
        count, unique = np.unique(Y_pred_sorted, return_counts=True)
        Y_enc = []
        for i, (c, u) in enumerate(
            sorted(zip(count, unique), key=lambda t: t[1], reverse=True)
        ):
            Y_enc += [i + 1 for _ in range(0, u)]

        Y_enc = np.array(Y_enc)
        counts, unique = np.unique(Y_enc, return_counts=True)
        disagreement_score = sum([c * u for c, u in zip(counts, unique)])
        #  print(Y_pred, "\t -> \t", Y_enc, "\t: ", disagreement_score)
        return disagreement_score

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
                        size=self.NR_QUERIES_PER_ITERATION,
                        replace=False,
                    )
                )
        elif (
            INITIAL_BATCH_SAMPLING_METHOD == "furthest"
            or INITIAL_BATCH_SAMPLING_METHOD == "furthest_lab"
            or INITIAL_BATCH_SAMPLING_METHOD == "uncertainty"
            or INITIAL_BATCH_SAMPLING_METHOD == "predicted_unity"
        ):
            possible_batches = [
                np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.NR_QUERIES_PER_ITERATION,
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
            elif INITIAL_BATCH_SAMPLING_METHOD == "predicted_unity":
                metric_function = self._calculate_predicted_unity
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
        elif INITIAL_BATCH_SAMPLING_METHOD == "hybrid":
            possible_batches = [
                np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.NR_QUERIES_PER_ITERATION,
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
            ][: math.floor(SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST)]

            furthest_lab_index_batches = [
                x
                for _, x in sorted(
                    zip(
                        [
                            self._calculate_furthest_lab_metric(a)
                            for a in possible_batches
                        ],
                        possible_batches,
                    ),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][
                : math.floor(
                    SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB
                )
            ]

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
            ][: math.floor(SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT)]

            predicted_unity_index_batches = [
                x
                for _, x in sorted(
                    zip(
                        [self._calculate_predicted_unity(a) for a in possible_batches],
                        possible_batches,
                    ),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][: math.floor(SAMPLE_SIZE * self.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY)]

            index_batches = [
                tuple(i.tolist())
                for i in (
                    furthest_index_batches
                    + furthest_lab_index_batches
                    + uncertainty_index_batches
                    + predicted_unity_index_batches
                )
            ]
            index_batches = set(index_batches)

            # add some random batches as padding
            index_batches = [np.array(list(i)) for i in index_batches] + [
                np.array(i)
                for i in random.sample(
                    set([tuple(i.tolist()) for i in possible_batches]).difference(
                        index_batches
                    ),
                    SAMPLE_SIZE - len(index_batches),
                )
            ]
        else:
            print(
                "NON EXISTENT INITIAL_SAMPLING_METHOD: " + INITIAL_BATCH_SAMPLING_METHOD
            )
            raise ()
        return index_batches

    def _get_normalized_unity_encoding_mapping(self):
        # adopted from https://stackoverflow.com/a/44209393
        def partitions(n, I=1):
            yield (n,)
            for i in range(I, n // 2 + 1):
                for p in partitions(n - i, i):
                    yield (i,) + p

        BATCH_SIZE = self.NR_QUERIES_PER_ITERATION
        N_CLASSES = len(self.data_storage.label_encoder.classes_)

        if N_CLASSES >= BATCH_SIZE:
            N_CLASSES = BATCH_SIZE
        possible_lengths = set()

        for possible_partition in partitions(BATCH_SIZE):
            if len(possible_partition) <= N_CLASSES:
                possible_lengths.add(
                    sum(
                        [
                            c * u
                            for c, u in zip(
                                sorted(possible_partition, reverse=True),
                                range(1, len(possible_partition) + 1),
                            )
                        ]
                    )
                )
        mapping = {}
        for i, possible_length in enumerate(sorted(possible_lengths)):
            mapping[possible_length] = i / (len(possible_lengths) - 1)
        return mapping

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        self.calculate_next_query_indices_pre_hook()
        batch_indices = self.get_X_query_index()

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.X_query_index = batch_indices
        X_state = self.calculate_state(
            batch_indices,
        )

        self.calculate_next_query_indices_post_hook(X_state)
        return batch_indices[self.get_sorting(X_state).argmax()]

    def calculate_state(
        self,
        batch_indices,
    ):
        state_list = []
        if self.STATE_UNCERTAINTIES:
            # normalize by batch size
            state_list += [
                (self.NR_QUERIES_PER_ITERATION + self._calculate_uncertainty_metric(a))
                / self.NR_QUERIES_PER_ITERATION
                for a in batch_indices
            ]

        if self.STATE_DISTANCES:
            # normalize based on the assumption, that the whole vector space got first normalized to -1 to +1, then we can calculate the maximum possible distance like this:
            state_list += [
                self._calculate_furthest_metric(a)
                / (
                    2
                    * math.sqrt(np.shape(self.data_storage.X)[1])
                    * self.NR_QUERIES_PER_ITERATION
                )
                for a in batch_indices
            ]
        if self.STATE_DISTANCES_LAB:
            state_list += [
                self._calculate_furthest_lab_metric(a)
                / (
                    2
                    * math.sqrt(np.shape(self.data_storage.X)[1])
                    * self.NR_QUERIES_PER_ITERATION
                )
                for a in batch_indices
            ]
        if self.STATE_PREDICTED_UNITY:
            pred_unity_mapping = self._get_normalized_unity_encoding_mapping()
            # normalize in a super complicated fashion due to the encoding
            state_list += [
                pred_unity_mapping[self._calculate_predicted_unity(a)]
                for a in batch_indices
            ]

        return np.array(state_list)
