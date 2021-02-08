from typing import Any, List, Sequence

from sklearn.naive_bayes import BernoulliNB
from active_learning.dataStorage import DataStorage, FeatureList, IndiceMask
from active_learning.learner.standard import Learner
from active_learning.sampling_strategies.BaseSamplingStrategy import (
    BaseSamplingStrategy,
)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import numpy as np
from sklearn.metrics import pairwise_distances
import abc

State = np.ndarray


class LearnedBaseSampling(BaseSamplingStrategy):
    PRE_SAMPLING_METHOD: str
    PRE_SAMPLING_ARG: Any
    STATE_ARGSECOND_PROBAS: bool
    STATE_ARGTHIRD_PROBAS: bool
    STATE_DIFF_PROBAS: bool
    STATE_PREDICTED_CLASS: bool
    STATE_DISTANCES_LAB: bool
    STATE_DISTANCES_UNLAB: bool
    STATE_INCLUDE_NR_FEATURES: bool
    DISTANCE_METRIC: str

    def __init__(
        self,
        PRE_SAMPLING_METHOD: str,
        PRE_SAMPLING_ARG: Any,
        STATE_ARGSECOND_PROBAS: bool = False,
        STATE_ARGTHIRD_PROBAS: bool = False,
        STATE_DIFF_PROBAS: bool = False,
        STATE_PREDICTED_CLASS: bool = False,
        STATE_DISTANCES_LAB: bool = False,
        STATE_DISTANCES_UNLAB: bool = False,
        STATE_INCLUDE_NR_FEATURES: bool = False,
        DISTANCE_METRIC: str = "euclidean",
    ) -> None:
        super().__init__()

        self.PRE_SAMPLING_METHOD = PRE_SAMPLING_METHOD
        self.PRE_SAMPLING_ARG = PRE_SAMPLING_ARG
        self.STATE_ARGSECOND_PROBAS = STATE_ARGSECOND_PROBAS
        self.STATE_ARGTHIRD_PROBAS = STATE_ARGTHIRD_PROBAS
        self.STATE_DIFF_PROBAS = STATE_DIFF_PROBAS
        self.STATE_PREDICTED_CLASS = STATE_PREDICTED_CLASS
        self.STATE_DISTANCES_LAB = STATE_DISTANCES_LAB
        self.STATE_DISTANCES_UNLAB = STATE_DISTANCES_UNLAB
        self.STATE_INCLUDE_NR_FEATURES = STATE_INCLUDE_NR_FEATURES
        self.DISTANCE_METRIC = DISTANCE_METRIC

    def what_to_label_next(
        self, NR_QUERIES_PER_ITERATION: int, learner: Learner, data_storage: DataStorage
    ) -> IndiceMask:
        self.data_storage: DataStorage = data_storage
        self.learner: Learner = learner
        self.NR_QUERIES_PER_ITERATION: int = NR_QUERIES_PER_ITERATION

        X_query_index = self.get_X_query_index()

        X_state: State = self.calculate_state(
            data_storage.X[X_query_index]
        )
        
        self.calculate_next_query_indices_post_hook(X_state)

        # use the optimal values
        zero_to_one_values_and_index = list(
            zip(self.get_sorting(X_state), X_query_index)
        )
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return np.array(
            [
                v
                for _, v in ordered_list_of_possible_sample_indices[
                    :NR_QUERIES_PER_ITERATION
                ]
            ]
        )

    def pre_sample_potential_X_queries(
        self,
        AMOUNT_OF_PEAKED_OBJECTS: int,
    ) -> IndiceMask:
        if self.PRE_SAMPLING_METHOD == "random":
            X_query_index: IndiceMask = np.random.choice(
                self.data_storage.unlabeled_mask, size=AMOUNT_OF_PEAKED_OBJECTS, replace=False
            )
        elif self.PRE_SAMPLING_METHOD == "furthest":
            max_sum = 0
            X_query_index: IndiceMask = np.empty(0)
            for _ in range(0, self.PRE_SAMPLING_ARG):
                random_index: IndiceMask = np.random.choice(
                    self.data_storage.unlabeled_mask, size=AMOUNT_OF_PEAKED_OBJECTS, replace=False
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

        return X_query_index
   
    @abc.abstractmethod
    def get_X_query_index(self) -> IndiceMask:
        pass

    @abc.abstractmethod
    def calculate_next_query_indices_post_hook(self, X_state: State):
        pass

    @abc.abstractmethod
    def get_sorting(self, X_state: State) -> List[float]:
        pass

    def calculate_state(
        self, X_query: FeatureList, 
    ) -> State:
        possible_samples_probas = self.learner.predict_proba(X_query)

        sorted_probas = -np.sort(-possible_samples_probas, axis=1) # type: ignore
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
