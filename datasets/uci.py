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


def _load_uci(self):
    df = pd.read_csv(self.DATASETS_PATH + "uci_cleaned/" + self.DATASET_NAME + ".csv")

    # shuffle df
    df = df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

    self.synthetic_creation_args = {}
    self.synthetic_creation_args["n_classes"] = len(df["LABEL"].unique())

    Y = df["LABEL"].to_numpy()
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    return df.loc[:, df.columns != "LABEL"].to_numpy(), Y
