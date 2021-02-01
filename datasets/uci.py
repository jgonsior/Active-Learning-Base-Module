import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import jit
from scipy.sparse import lil_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler


def load_uci(DATASETS_PATH: str, DATASET_NAME: str, RANDOM_SEED: int) -> pd.DataFrame:
    df = pd.read_csv(DATASETS_PATH + "uci_cleaned/" + DATASET_NAME + ".csv")

    # shuffle df
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    synthetic_creation_args = {}
    synthetic_creation_args["n_classes"] = len(df["LABEL"].unique())

    Y = df["LABEL"].to_numpy()
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    df["LABEL"] = Y

    return df
