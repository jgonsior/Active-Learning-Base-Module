import os

import dill
import numpy as np

from .learnedBaseBatchSampling import LearnedBaseBatchSampling


class TrainedBatchNNLearner(LearnedBaseBatchSampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(kwargs["NN_BINARY_PATH"], "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

    def get_X_query_index(self):
        return self.sample_unlabeled_X(
            self.sampling_classifier.n_outputs_,
            INITIAL_BATCH_SAMPLING_ARG=self.INITIAL_BATCH_SAMPLING_ARG,
            INITIAL_BATCH_SAMPLING_METHOD=self.INITIAL_BATCH_SAMPLING_METHOD,
        )

    def get_sorting(self, X_state):
        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state)
        sorting = Y_pred
        return sorting
