from typing import Any, List
from active_learning.dataStorage import IndiceMask, LabelList
import random
import math
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import copy

import numpy as np
from sklearn.metrics import pairwise_distances
from .learnedBaseSampling import LearnedBaseSampling


class LearnedBaseBatchSampling(LearnedBaseSampling):
    STATE_PREDICTED_UNITY: bool
    STATE_DISTANCES: bool
    STATE_UNCERTAINTIES: bool
    PRE_SAMPLING_BATCH_FRACTION_HYBRID_UNCERT: float
    PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST: float
    PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST_LAB: float
    PRE_SAMPLING_BATCH_FRACTION_HYBRID_PRED_UNITY: float

    def __init__(
        self,
        PRE_SAMPLING_METHOD: str,
        PRE_SAMPLING_ARG: Any,
        DISTANCE_METRIC: str = "euclidean",
        STATE_PREDICTED_UNITY: bool = False,
        STATE_DISTANCES: bool = True,
        STATE_UNCERTAINTIES: bool = True,
        PRE_SAMPLING_BATCH_FRACTION_HYBRID_UNCERT: float = 0.33,
        PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST: float = 0.33,
        PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST_LAB: float = 0,
        PRE_SAMPLING_BATCH_FRACTION_HYBRID_PRED_UNITY: float = 0.33,
    ) -> None:
        super().__init__(
            PRE_SAMPLING_METHOD=PRE_SAMPLING_METHOD,
            PRE_SAMPLING_ARG=PRE_SAMPLING_ARG,
            DISTANCE_METRIC=DISTANCE_METRIC,
        )

        self.STATE_PREDICTED_UNITY = STATE_PREDICTED_UNITY
        self.STATE_DISTANCES = STATE_DISTANCES
        self.STATE_UNCERTAINTIES = STATE_UNCERTAINTIES
        self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_UNCERT = (
            PRE_SAMPLING_BATCH_FRACTION_HYBRID_UNCERT
        )
        self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST = (
            PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST
        )
        self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST_LAB = (
            PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST_LAB
        )
        self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_PRED_UNITY = (
            PRE_SAMPLING_BATCH_FRACTION_HYBRID_PRED_UNITY
        )

    def _calculate_furthest_metric(self, batch_indices: IndiceMask) -> float:
        return np.sum(
            pairwise_distances(
                self.data_storage.X[batch_indices],
                self.data_storage.X[batch_indices],
                metric=self.DISTANCE_METRIC,
            )
        )

    def _calculate_furthest_lab_metric(self, batch_indices: IndiceMask) -> float:
        return np.sum(
            pairwise_distances(
                self.data_storage.X[batch_indices],
                self.data_storage.X[self.data_storage.labeled_mask],
                metric=self.DISTANCE_METRIC,
            )
        )

    def _calculate_uncertainty_metric(self, batch_indices: IndiceMask) -> float:
        Y_proba = self.Y_probas[batch_indices]
        #  Y_proba = self.clf.predict_proba(self.data_storage.X[batch_indices])
        margin = np.partition(-Y_proba, 1, axis=1)  # type: ignore
        return np.sum(-np.abs(margin[:, 0] - margin[:, 1]))

    def _calculate_predicted_unity(self, unlabeled_sample_indices: IndiceMask) -> float:
        Y_pred: LabelList = self.Y_preds[unlabeled_sample_indices]
        #  Y_pred = self.clf.predict(self.data_storage.X[unlabeled_sample_indices])
        Y_pred_sorted: LabelList = np.sort(Y_pred)
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

    def pre_sample_potential_X_queries(
        self,
        AMOUNT_OF_PEAKED_OBJECTS: int,
    ) -> IndiceMask:
        index_batches = []
        possible_batches: List[IndiceMask]

        if self.PRE_SAMPLING_METHOD == "random":
            for _ in range(0, AMOUNT_OF_PEAKED_OBJECTS):
                index_batches.append(
                    np.random.choice(
                        self.data_storage.unlabeled_mask,
                        size=self.NR_QUERIES_PER_ITERATION,
                        replace=False,
                    )
                )
        elif (
            self.PRE_SAMPLING_METHOD == "furthest"
            or self.PRE_SAMPLING_METHOD == "furthest_lab"
            or self.PRE_SAMPLING_METHOD == "uncertainty"
            or self.PRE_SAMPLING_METHOD == "predicted_unity"
        ):
            possible_batches = [
                np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.NR_QUERIES_PER_ITERATION,
                    replace=False,
                )
                for _ in range(0, self.PRE_SAMPLING_ARG)
            ]

            if self.PRE_SAMPLING_METHOD == "furthest":
                metric_function = self._calculate_furthest_metric
            elif self.PRE_SAMPLING_METHOD == "furthest_lab":
                metric_function = self._calculate_furthest_lab_metric
            elif self.PRE_SAMPLING_METHOD == "uncertainty":
                metric_function = self._calculate_uncertainty_metric
            elif self.PRE_SAMPLING_METHOD == "predicted_unity":
                metric_function = self._calculate_predicted_unity
            else:
                print("The defined PRE_SAMPLING_METHOD does not exist, exiting…")
                exit(-1)
            metric_values = [metric_function(a) for a in possible_batches]

            # take n samples based on the sorting metric, the rest randomly
            index_batches = [
                x
                for _, x in sorted(
                    zip(metric_values, possible_batches),
                    key=lambda t: t[0],
                    reverse=True,
                )
            ][:AMOUNT_OF_PEAKED_OBJECTS]
        elif self.PRE_SAMPLING_METHOD == "hybrid":
            possible_batches = [
                np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.NR_QUERIES_PER_ITERATION,
                    replace=False,
                )
                for _ in range(0, self.PRE_SAMPLING_ARG)
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
            ][
                : math.floor(
                    AMOUNT_OF_PEAKED_OBJECTS
                    * self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST
                )
            ]

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
                    AMOUNT_OF_PEAKED_OBJECTS
                    * self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_FURTHEST_LAB
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
            ][
                : math.floor(
                    AMOUNT_OF_PEAKED_OBJECTS
                    * self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_UNCERT
                )
            ]

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
            ][
                : math.floor(
                    AMOUNT_OF_PEAKED_OBJECTS
                    * self.PRE_SAMPLING_BATCH_FRACTION_HYBRID_PRED_UNITY
                )
            ]

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
                    AMOUNT_OF_PEAKED_OBJECTS - len(index_batches),
                )
            ]
        else:
            print("NON EXISTENT INITIAL_SAMPLING_METHOD: " + self.PRE_SAMPLING_METHOD)
            exit(-1)
        return np.array(list(index_batches))

    def _get_normalized_unity_encoding_mapping(self):
        # adopted from https://stackoverflow.com/a/44209393
        def partitions(n, I=1):
            yield (n,)
            for i in range(I, n // 2 + 1):
                for p in partitions(n - i, i):
                    yield (i,) + p

        BATCH_SIZE = self.NR_QUERIES_PER_ITERATION
        N_CLASSES = len(self.data_storage.label_encoder.classes_)  # type: ignore

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
        self.Y_probas = self.learner.predict_proba(self.data_storage.X)
        self.Y_preds = self.learner.predict(self.data_storage.X)

        batch_indices = self.get_X_query_index()

        X_state = self.calculate_state(
            batch_indices,
        )

        self.calculate_next_query_indices_post_hook(X_state)
        return batch_indices[np.argmax(self.get_sorting(X_state))]

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
            if self.DISTANCE_METRIC == "euclidean":
                normalization_denominator = (
                    2
                    * math.sqrt(self.data_storage.X.shape[1])
                    * self.NR_QUERIES_PER_ITERATION
                )
            elif self.DISTANCE_METRIC == "cosine":
                normalization_denominator = self.NR_QUERIES_PER_ITERATION
            else:
                print("The defined distance metric is not implemented, exiting…")
                exit(-1)
            state_list += [
                self._calculate_furthest_metric(a) / normalization_denominator
                for a in batch_indices
            ]
        if self.STATE_DISTANCES_LAB:
            if self.DISTANCE_METRIC == "euclidean":
                normalization_denominator = (
                    2
                    * math.sqrt(self.data_storage.X.shape[1])
                    * self.NR_QUERIES_PER_ITERATION
                )
            elif self.DISTANCE_METRIC == "cosine":
                normalization_denominator = self.NR_QUERIES_PER_ITERATION
            else:
                print("The defined distance metric is not implemented, exiting…")
                exit(-1)
            state_list += [
                self._calculate_furthest_lab_metric(a) / (normalization_denominator)
                for a in batch_indices
            ]
        if self.STATE_PREDICTED_UNITY:
            pred_unity_mapping = self._get_normalized_unity_encoding_mapping()
            # normalize in a super complicated fashion due to the encoding
            state_list += [
                pred_unity_mapping[self._calculate_predicted_unity(a)]
                for a in batch_indices
            ]

        if self.STATE_INCLUDE_NR_FEATURES:
            state_list = [self.data_storage.X.shape[1]] + state_list
        return np.array(state_list)
