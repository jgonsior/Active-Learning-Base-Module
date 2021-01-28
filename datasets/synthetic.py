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


def old_synhtetic(self):
    pass


def new_synthetic(self):
    pass


def _load_synthetic(self):
    no_valid_synthetic_arguments_found = True
    while no_valid_synthetic_arguments_found:
        if not self.NEW_SYNTHETIC_PARAMS:
            if self.VARIABLE_DATASET:
                N_SAMPLES = random.randint(100, 5000)
            else:
                N_SAMPLES = random.randint(100, 2000)

            if self.AMOUNT_OF_FEATURES > 0:
                N_FEATURES = self.AMOUNT_OF_FEATURES
            else:
                N_FEATURES = random.randint(2, 100)

            N_INFORMATIVE, N_REDUNDANT, N_REPEATED = [
                int(N_FEATURES * i)
                for i in np.random.dirichlet(np.ones(3), size=1).tolist()[0]
            ]

            N_CLASSES = random.randint(2, 10)
            N_CLUSTERS_PER_CLASS = random.randint(
                1, min(max(1, int(2 ** N_INFORMATIVE / N_CLASSES)), 10)
            )

            if N_CLASSES * N_CLUSTERS_PER_CLASS > 2 ** N_INFORMATIVE:
                continue
            no_valid_synthetic_arguments_found = False

            WEIGHTS = np.random.dirichlet(np.ones(N_CLASSES), size=1).tolist()[
                0
            ]  # list of weights, len(WEIGHTS) = N_CLASSES, sum(WEIGHTS)=1

            if self.GENERATE_NOISE:
                FLIP_Y = (
                    np.random.pareto(2.0) + 1
                ) * 0.01  # amount of noise, larger values make it harder
            else:
                FLIP_Y = 0

            CLASS_SEP = random.uniform(
                0, 10
            )  # larger values spread out the clusters and make it easier
            HYPERCUBE = self.HYPERCUBE  # if false random polytope
            SCALE = 0.01  # features should be between 0 and 1 now
        else:
            if self.VARIABLE_DATASET:
                N_SAMPLES = random.randint(100, 5000)
            else:
                N_SAMPLES = random.randint(100, 2000)
                #  N_SAMPLES = 1000
            if self.AMOUNT_OF_FEATURES > 0:
                N_FEATURES = self.AMOUNT_OF_FEATURES
            else:
                N_FEATURES = random.randint(2, 20)
            N_REDUNDANT = N_REPEATED = 0
            N_INFORMATIVE = N_FEATURES

            N_CLASSES = random.randint(2, min(10, 2 ** N_INFORMATIVE))
            N_CLUSTERS_PER_CLASS = random.randint(
                1, int(2 ** N_INFORMATIVE / N_CLASSES)
            )

            if N_CLASSES * N_CLUSTERS_PER_CLASS > 2 ** N_INFORMATIVE:
                print("ui")
                continue
            no_valid_synthetic_arguments_found = False

            WEIGHTS = np.random.dirichlet(np.ones(N_CLASSES), size=1).tolist()[
                0
            ]  # list of weights, len(WEIGHTS) = N_CLASSES, sum(WEIGHTS)=1

            if self.GENERATE_NOISE:
                FLIP_Y = (
                    np.random.pareto(2.0) + 1
                ) * 0.01  # amount of noise, larger values make it harder
            else:
                FLIP_Y = 0

            CLASS_SEP = random.uniform(
                0, 10
            )  # larger values spread out the clusters and make it easier

            HYPERCUBE = self.HYPERCUBE  # if false a random polytope is selected instead
            SCALE = 0.01  # features should be between 0 and 1 now

        synthetic_creation_args = {
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_informative": N_INFORMATIVE,
            "n_redundant": N_REDUNDANT,
            "n_repeated": N_REPEATED,
            "n_classes": N_CLASSES,
            "n_clusters_per_class": N_CLUSTERS_PER_CLASS,
            "weights": WEIGHTS,
            "flip_y": FLIP_Y,
            "class_sep": CLASS_SEP,
            "hypercube": HYPERCUBE,
            "scale": SCALE,
            "random_state": self.RANDOM_SEED,
        }
        self.synthetic_creation_args = synthetic_creation_args

    log_it(synthetic_creation_args)
    X, Y = make_classification(**synthetic_creation_args)

    return X, Y
