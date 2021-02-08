from typing import List
from active_learning.dataStorage import IndiceMask
import os

import dill
import numpy as np

from .learnedBaseSampling import LearnedBaseSampling, State


class TrainedNNLearner(LearnedBaseSampling):
    def __init__(self, NN_BINARY_PATH:str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

    def get_X_query_index(self) -> IndiceMask:
        return self.pre_sample_potential_X_queries(
            self.sampling_classifier.n_outputs_,
        )

    def get_sorting(self, X_state: State)->List[float]:
        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state)
        sorting = Y_pred
        return sorting
