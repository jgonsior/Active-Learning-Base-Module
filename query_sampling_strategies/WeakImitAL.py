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

    """
        möglicher input:
        pre-sampling -> verschiedene methoden ausprobieren(
            distance among points
            uncert von weak labellern uneinig
            hat schon label von weak labellern erhalten, aber zugleich ist vorhersage dafür vom Netz sehr gering (trotz enthalten im trainingsset!!)

        pro sample:
        - most confident weak labeller prediction
        - "disagreement score" von weak labellern -> enthält vorheriges vielleicht sogar?
        - average distance to unlabeld/labeleled/human_labelled points

        output:
        pro sample:
        - task annotation by human label
        - take annotation by weak labeller

        Erroneous labels:
        easy test: just treat weakly labeled samples as "unlabeled" by the main classifier again!!
    """

    def encode_input_state(
        self, pre_sampled_X_queries_indices: PreSampledIndices
    ) -> InputState:
        state_list: List[float] = []
        # per sample:

        X_query = self.data_storage.X[pre_sampled_X_queries_indices]

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
