from sklearn.metrics.pairwise import pairwise_distances
from active_learning.sampling_strategies.ImitationLearner import TrainImitALSingle
import copy
from typing import List, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from .BatchStateEncoding import BatchStateSampling

from active_learning.dataStorage import IndiceMask

from .ImitationLearningBaseSampling import (
    ImitationLearningBaseSampling,
    InputState,
    OutputState,
    PreSampledIndices,
)
from .SingleStateEncoding import SingleStateEncoding


class WeakImitAL(TrainImitALSingle):
    STATE_ARGSECOND_PROBAS: bool
    STATE_ARGTHIRD_PROBAS: bool
    STATE_DIFF_PROBAS: bool
    STATE_PREDICTED_CLASS: bool
    STATE_DISTANCES_LAB: bool
    STATE_DISTANCES_UNLAB: bool
    STATE_INCLUDE_NR_FEATURES: bool

    """
        möglicher input:
        pro sample:
        - most confident weak labeller prediction
        - average distance to unlabeld/labeleled/human_labelled points

        output:
        pro sample:
        - task annotation by human label
        - take annotation by weak labeller

        einfacher test: es gibt automatische weak labeller (einfache synthetische annotation functions), mein netz wird darauf trainiert (mit imit learning), nur dann den user zu fragen, wenn die labelling functions falsch liegen?
        --> ich muss synthetische labelling functions so gestalten, dass sie auch tatsächlich ab und zu falsch sind

        Erroneous labels:
        easy test: just treat weakly labeled samples as "unlabeled" by the main classifier again!!
    """

    def __init__(
        self,
        STATE_ARGSECOND_PROBAS: bool = False,
        STATE_ARGTHIRD_PROBAS: bool = False,
        STATE_DIFF_PROBAS: bool = False,
        STATE_PREDICTED_CLASS: bool = False,
        STATE_DISTANCES_LAB: bool = False,
        STATE_DISTANCES_UNLAB: bool = False,
        STATE_INCLUDE_NR_FEATURES: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.STATE_ARGSECOND_PROBAS = STATE_ARGSECOND_PROBAS
        self.STATE_ARGTHIRD_PROBAS = STATE_ARGTHIRD_PROBAS
        self.STATE_DIFF_PROBAS = STATE_DIFF_PROBAS
        self.STATE_PREDICTED_CLASS = STATE_PREDICTED_CLASS
        self.STATE_DISTANCES_LAB = STATE_DISTANCES_LAB
        self.STATE_DISTANCES_UNLAB = STATE_DISTANCES_UNLAB
        self.STATE_INCLUDE_NR_FEATURES = STATE_INCLUDE_NR_FEATURES

    def calculateImitationLearningData(
        self, pre_sampled_X_querie_indices: PreSampledIndices
    ) -> None:
        future_peak_acc = []
        # single thread
        for unlabeled_sample_indices in pre_sampled_X_querie_indices:
            future_peak_acc.append(
                self._future_peak(np.array([unlabeled_sample_indices]))
            )

        self.optimal_policies = self.optimal_policies.append(
            pd.Series(dict(zip(self.optimal_policies.columns, future_peak_acc))),  # type: ignore
            ignore_index=True,
        )

    def pre_sample_potential_X_queries(
        self,
    ) -> PreSampledIndices:
        if self.PRE_SAMPLING_METHOD == "random":
            X_query_index: PreSampledIndices = np.random.choice(
                self.data_storage.unlabeled_mask,
                size=self.AMOUNT_OF_PEAKED_OBJECTS,
                replace=False,
            )
        elif self.PRE_SAMPLING_METHOD == "furthest":
            max_sum = 0
            X_query_index: PreSampledIndices = np.empty(0, dtype=np.int64)
            for _ in range(0, self.PRE_SAMPLING_ARG):
                random_index: PreSampledIndices = np.random.choice(
                    self.data_storage.unlabeled_mask,
                    size=self.AMOUNT_OF_PEAKED_OBJECTS,
                    replace=False,
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
                + self.PRE_SAMPLING_METHOD
            )
            exit(-1)

        # return np.reshape(X_query_index,(1, len(X_query_index))) # type: ignore
        return X_query_index

    def encode_input_state(
        self, pre_sampled_X_queries_indices: PreSampledIndices
    ) -> InputState:
        X_query = self.data_storage.X[pre_sampled_X_queries_indices]
        possible_samples_probas = self.learner.predict_proba(X_query)

        sorted_probas = -np.sort(-possible_samples_probas, axis=1)  # type: ignore
        argmax_probas = sorted_probas[:, 0]

        state_list: List[float] = list(argmax_probas.tolist())

        if self.STATE_ARGSECOND_PROBAS:
            argsecond_probas = sorted_probas[:, 1]
            state_list += argsecond_probas.tolist()
        if self.STATE_DIFF_PROBAS:
            state_list += (argmax_probas - sorted_probas[:, 1]).tolist()
        if self.STATE_ARGTHIRD_PROBAS:
            if sorted_probas.shape[1] < 3:
                state_list += [0 for _ in range(0, len(X_query))]
            else:
                state_list += sorted_probas[:, 2].tolist()
        if self.STATE_PREDICTED_CLASS:
            state_list += self.learner.predict(X_query).tolist()

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
            state_list = [float(self.data_storage.X.shape[1])] + state_list
        return np.array(state_list)

    def _future_peak(
        self,
        unlabeled_sample_indices: IndiceMask,
    ) -> float:
        copy_of_classifier = copy.deepcopy(self.learner)

        copy_of_labeled_mask = np.append(
            self.data_storage.labeled_mask, unlabeled_sample_indices, axis=0
        )

        copy_of_classifier.fit(
            self.data_storage.X[copy_of_labeled_mask],
            self.data_storage.Y[copy_of_labeled_mask],
        )

        Y_pred_test = copy_of_classifier.predict(self.data_storage.X)
        Y_true = self.data_storage.Y

        accuracy_with_that_label = accuracy_score(Y_pred_test, Y_true)

        return accuracy_with_that_label
