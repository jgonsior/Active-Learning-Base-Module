import abc
import os
from typing import Any

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from active_learning.dataStorage import DataStorage, FeatureList, IndiceMask
from active_learning.learner.standard import Learner
from active_learning.sampling_strategies.BaseSamplingStrategy import (
    BaseSamplingStrategy,
)
from train_lstm import AMOUNT_OF_PEAKED_OBJECTS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

InputState = np.ndarray[np.float64]
OutputState = np.ndarray[np.float64]
PreSampledIndices = (
    np.ndarray
)  # these are lists of indices, for batch batches, and for single just one-element lists


class ImitationLearningBaseSampling(BaseSamplingStrategy):
    PRE_SAMPLING_METHOD: str
    PRE_SAMPLING_ARG: Any
    DISTANCE_METRIC: str
    AMOUNT_OF_PEAKED_OBJECTS: int

    def __init__(
        self,
        PRE_SAMPLING_METHOD: str,
        PRE_SAMPLING_ARG: Any,
        AMOUNT_OF_PEAKED_OBJECTS: int,
        DISTANCE_METRIC: str = "euclidean",
    ) -> None:
        super().__init__()

        self.PRE_SAMPLING_METHOD = PRE_SAMPLING_METHOD
        self.PRE_SAMPLING_ARG = PRE_SAMPLING_ARG
        self.AMOUNT_OF_PEAKED_OBJECTS = AMOUNT_OF_PEAKED_OBJECTS

        self.DISTANCE_METRIC = DISTANCE_METRIC

    def what_to_label_next(
        self, BATCH_SIZE: int, learner: Learner, data_storage: DataStorage
    ) -> IndiceMask:
        self.data_storage: DataStorage = data_storage
        self.learner: Learner = learner
        self.NR_QUERIES_PER_ITERATION: int = BATCH_SIZE

        pre_sampled_X_querie_indices: PreSampledIndices = (
            self.pre_sample_potential_X_queries()
        )

        # when using a pre-trained model this does nothing
        self.calculateImitationLearningData(pre_sampled_X_querie_indices)

        X_input_state: InputState = self.encode_input_state(
            pre_sampled_X_querie_indices
        )
        Y_output_state: OutputState = self.applyNN(X_input_state)
        return self.decode_output_state(
            Y_output_state, pre_sampled_X_querie_indices, BATCH_SIZE
        )

    @abc.abstractmethod
    def pre_sample_potential_X_queries(self) -> PreSampledIndices:
        pass

    """ This method uses when training, a "peak into the future" to mimic a greedy, but potentially good expert action
        When using a trained NN we simply apply the network to the samples
    """

    @abc.abstractmethod
    def calculateImitationLearningData(
        self, pre_sampled_X_querie_indices: PreSampledIndices
    ) -> None:
        pass

    @abc.abstractmethod
    def applyNN(self, X_input_state: InputState) -> OutputState:
        pass

    @abc.abstractmethod
    def encode_input_state(
        self, pre_sampled_X_querie_indices: PreSampledIndices
    ) -> InputState:
        pass

    def decode_output_state(
        self,
        Y_output_state: OutputState,
        pre_sampled_X_querie_indices: PreSampledIndices,
        BATCH_SIZE: int,
    ) -> IndiceMask:
        zero_to_one_values_and_index = list(
            zip(Y_output_state, pre_sampled_X_querie_indices)
        )
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return np.array(
            [v for _, v in ordered_list_of_possible_sample_indices[:BATCH_SIZE]]
        )
