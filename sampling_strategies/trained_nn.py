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
        REPRESENTATIVE_FEATURES,
        CONVEX_HULL_SAMPLING,
        NO_DIFF_FEATURES,
        LRU_AREAS_LIMIT,
    ):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

        super().init_sampling_classifier(
            LRU_AREAS_LIMIT=LRU_AREAS_LIMIT,
            NO_DIFF_FEATURES=NO_DIFF_FEATURES,
            CONVEX_HULL_SAMPLING=CONVEX_HULL_SAMPLING,
            REPRESENTATIVE_FEATURES=REPRESENTATIVE_FEATURES,
        )

    def get_X_query(self):
        return self.sample_unlabeled_X(
            self.data_storage.train_unlabeled_X,
            self.data_storage.train_labeled_X,
            self.sampling_classifier.n_outputs_,
            self.CONVEX_HULL_SAMPLING,
        )

    def get_sorting(self, X_state):
        X_state = np.reshape(X_state, (1, len(X_state)))
        Y_pred = self.sampling_classifier.predict(X_state)
        sorting = Y_pred
        return sorting
