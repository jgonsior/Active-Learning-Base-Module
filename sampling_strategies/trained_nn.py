import pandas as pd
import os
from sklearn.metrics import pairwise_distances
import random
from itertools import chain

import dill
import numpy as np

from .learnedBaseSampling import LearnedBaseSampling


class TrainedNNLearner(LearnedBaseSampling):
    def init_sampling_classifier(
        self,
        NN_BINARY_PATH,
        STATE_DISTANCES_LAB,
        STATE_DISTANCES_UNLAB,
        STATE_DIFF_PROBAS,
        STATE_ARGTHIRD_PROBAS,
        STATE_PREDICTED_CLASS,
        STATE_ARGSECOND_PROBAS,
        INITIAL_BATCH_SAMPLING_METHOD,
        INITIAL_BATCH_SAMPLING_ARG,
    ):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

        super().init_sampling_classifier(
            STATE_DISTANCES_LAB=STATE_DISTANCES_LAB,
            STATE_DISTANCES_UNLAB=STATE_DISTANCES_UNLAB,
            STATE_DIFF_PROBAS=STATE_DIFF_PROBAS,
            STATE_ARGTHIRD_PROBAS=STATE_ARGTHIRD_PROBAS,
            STATE_ARGSECOND_PROBAS=STATE_ARGSECOND_PROBAS,
            STATE_PREDICTED_CLASS=STATE_PREDICTED_CLASS,
            INITIAL_BATCH_SAMPLING_ARG=INITIAL_BATCH_SAMPLING_ARG,
            INITIAL_BATCH_SAMPLING_METHOD=INITIAL_BATCH_SAMPLING_METHOD,
        )

    def get_X_query_index(self):
        return self.sample_unlabeled_X(
            self.sampling_classifier.n_outputs_,
            INITIAL_BATCH_SAMPLING_ARG=self.INITIAL_BATCH_SAMPLING_ARG,
            INITIAL_BATCH_SAMPLING_METHOD=self.INITIAL_BATCH_SAMPLING_METHOD,
        )

    def get_sorting(self, X_state):
        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state, verbose=0)
        sorting = Y_pred
        return sorting
