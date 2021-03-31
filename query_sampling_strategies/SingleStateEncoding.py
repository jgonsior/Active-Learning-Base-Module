from .ImitationLearningBaseQuerySampler import (
    ImitationLearningBaseQuerySampler,
    InputState,
    PreSampledIndices,
)
from typing import List

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class SingleStateEncoding(ImitationLearningBaseQuerySampler):
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
