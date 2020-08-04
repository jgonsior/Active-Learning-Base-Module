import dill
from scipy.stats import spearmanr, kendalltau
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed, parallel_backend
import random
from itertools import chain
import copy
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, mean_squared_error
from ..activeLearner import ActiveLearner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


class TrainedNNLearner(ActiveLearner):
    def init_sampling_classifier(self, NN_BINARY_PATH):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        with open(NN_BINARY_PATH, "rb") as handle:
            model = dill.load(handle)

        self.sampling_classifier = model

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        train_unlabeled_X_indices = list(
            chain(*list(train_unlabeled_X_cluster_indices.values()))
        )

        # do this n couple of times to make out of the semi pairwise a real listwise???

        zero_to_one_values_and_index = []
        for _ in range(0, 5):
            random.shuffle(train_unlabeled_X_indices)
            possible_samples_indices = train_unlabeled_X_indices[
                : self.sampling_classifier.n_outputs_
            ]

            possible_samples_probas = self.clf.predict_proba(
                self.data_storage.train_unlabeled_X.loc[possible_samples_indices]
            )

            sorted_probas = -np.sort(-possible_samples_probas, axis=1)
            argmax_probas = sorted_probas[:, 0]
            argsecond_probas = sorted_probas[:, 1]

            X_state = np.array([*argmax_probas, *argsecond_probas])
            X_state = np.reshape(X_state, (1, len(X_state)))
            Y_pred = self.sampling_classifier.predict(X_state)

            sorting = Y_pred

            zero_to_one_values_and_index += list(zip(sorting, possible_samples_indices))

        #  print(zero_to_one_values_and_index)
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return [
            v
            for k, v in ordered_list_of_possible_sample_indices[
                : self.nr_queries_per_iteration
            ]
        ]
