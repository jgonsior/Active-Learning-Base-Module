import os
from typing import List

import dill
import numpy as np

from active_learning.dataStorage import IndiceMask

from .BatchStateEncoding import BatchStateSampling
from .ImitationLearningBaseSampling import (ImitationLearningBaseSampling,
                                            InputState, OutputState)
from .SingleStateEncoding import SingleStateEncoding


class TrainedImitALSampling(ImitationLearningBaseSampling):
    def __init__(self, NN_BINARY_PATH: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

    def applyNN(self, X_input_state: InputState) -> OutputState:
        X_state = np.reshape(X_input_state, (1, len(X_input_state)))
        Y_pred: OutputState = self.sampling_classifier.predict(X_state)
        return Y_pred


class TrainedImitALSingleSampling(TrainedImitALSampling, SingleStateEncoding):
    pass


class TrainedImitALBatchSampling(TrainedImitALSampling, BatchStateSampling):
    pass
