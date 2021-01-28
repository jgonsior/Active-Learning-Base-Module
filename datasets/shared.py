
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from numba import jit
from scipy.sparse import lil_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from .experiment_setup_lib import log_it
def prepare_dataset():
    if df is None:
            # experiment
            log_it("Loading " + self.DATASET_NAME)
            if self.DATASET_NAME == "dwtc":
                X, Y = self._load_dwtc()
            elif self.DATASET_NAME == "synthetic":
                X, Y = self._load_synthetic()
            else:
                X, Y = self._load_uci()
                #  df = self._load_alc(DATASET_NAME, DATASETS_PATH)
        else:
            # real data
            X = df.loc[:, df.columns != "label"]
            Y = df["label"].to_numpy().reshape(len(X))
        #  X, Y = self._load_synthetic()
        #  df = None
        self.label_encoder = LabelEncoder()
        # feature normalization
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

        # scale back to [0,1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        self.label_source = np.full(len(Y), "N", dtype=str)
        self.X = X

        # check if we are in an experiment setting or are dealing with real, unlabeled data
        if df is not None:
            self.unlabeled_mask = np.argwhere(pd.isnull(Y)).flatten()
            self.labeled_mask = np.argwhere(~pd.isnull(Y)).flatten()
            self.label_source[self.labeled_mask] = ["G" for _ in self.labeled_mask]
            #  self.Y = Y

            # create test split out of labeled data
            self.test_mask = []  # self.labeled_mask[
            #                0 : math.floor(len(self.labeled_mask) * self.TEST_FRACTION)
            #           ]
            #  self.labeled_mask = self.labeled_mask[
            #      math.floor(len(self.labeled_mask) * self.TEST_FRACTION) :
            #  ]
            #  self.labeled_mask = np.empty(0, dtype=np.int64)

            Y_encoded = self.label_encoder.fit_transform(Y[~pd.isnull(Y)])
            self.Y = Y
            self.Y[self.labeled_mask] = Y_encoded

            self.Y[pd.isnull(self.Y)] = -1
            self.Y = self.Y.astype("int64")

        else:
            # ignore nan as labels
            Y = self.label_encoder.fit_transform(Y[~np.isnan(Y)])

            # split into test, train_labeled, train_unlabeled
            # experiment setting apparently

            self.unlabeled_mask = np.arange(
                math.floor(len(Y) * self.TEST_FRACTION), len(Y)
            )

            # prevent that the first split contains not all labels in the training split, so we just shuffle the data as long as we have every label in their
            while len(np.unique(Y[self.unlabeled_mask])) != len(
                self.label_encoder.classes_
            ):
                new_shuffled_indices = np.random.permutation(len(Y))
                X = X[new_shuffled_indices]
                Y = Y[new_shuffled_indices]
                self.unlabeled_mask = np.arange(
                    math.floor(len(Y) * self.TEST_FRACTION), len(Y)
                )
            self.test_mask = np.arange(0, math.floor(len(Y) * self.TEST_FRACTION))
            self.labeled_mask = np.empty(0, dtype=np.int64)
            self.Y = Y

            """ 
            1. get start_set from X_labeled
            2. if X_unlabeled is None : experiment!
                2.1 if X_test: rest von X_labeled wird X_train_unlabeled
                2.2 if X_test is none: split rest von X_labeled in X_train_unlabeled und X_test
               else (kein experiment):
               X_unlabeled wird X_unlabeled, rest von X_labeled wird X_train_unlabeled_
            
            """
            # separate X_labeled into start_set and labeled _rest
            # check if the minimum amount of labeled data is present in the start set size
            labels_not_in_start_set = set(range(0, len(self.label_encoder.classes_)))
            all_label_in_start_set = False

            if not all_label_in_start_set:
                #  if len(self.train_labeled_data) == 0:
                #      print("Please specify at least one labeled example of each class")
                #      exit(-1)

                # move more data here from the classes not present
                for label in labels_not_in_start_set:
                    # select a random sample of this labelwhich is NOT yet labeled
                    random_index = np.where(self.Y[self.unlabeled_mask] == label)[0][0]

                    # the random_index before is an index on Y[unlabeled_mask], and therefore NOT the same as an index on purely Y
                    # therefore it needs to be converted first
                    random_index = self.unlabeled_mask[random_index]

                    self._label_samples_without_clusters([random_index], label, "G")

