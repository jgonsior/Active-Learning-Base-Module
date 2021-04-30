import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from typing import Any, Dict, Tuple

from active_learning.logger.logger import log_it


def load_synthetic(
    RANDOM_SEED: int,
    NEW_SYNTHETIC_PARAMS: bool,
    VARIABLE_DATASET: bool,
    AMOUNT_OF_FEATURES: int,
    HYPERCUBE: bool,
    GENERATE_NOISE: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    no_valid_synthetic_arguments_found = True
    while no_valid_synthetic_arguments_found:
        log_it("Start finding synthetic dataset")
        N_SAMPLES = random.randint(100, 5000)
        N_FEATURES = random.randint(2, 100)

        N_INFORMATIVE, N_REDUNDANT, N_REPEATED = [
            int(N_FEATURES * i)
            for i in np.random.dirichlet(np.ones(3), size=1).tolist()[0]  # type: ignore
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

        FLIP_Y = (
            np.random.pareto(2.0) + 1  # type: ignore
        ) * 0.01  # amount of noise, larger values make it harder

        CLASS_SEP = random.uniform(
            0, 10
        )  # larger values spread out the clusters and make it easier
        HYPERCUBE = False  # if false random polytope
        SCALE = 0.01  # features should be between 0 and 1 now

    synthetic_creation_args = {
        "n_samples": N_SAMPLES,  # type: ignore
        "n_features": N_FEATURES,  # type: ignore
        "n_informative": N_INFORMATIVE,  # type: ignore
        "n_redundant": N_REDUNDANT,  # type: ignore
        "n_repeated": N_REPEATED,  # type: ignore
        "n_classes": N_CLASSES,  # type: ignore
        "n_clusters_per_class": N_CLUSTERS_PER_CLASS,  # type: ignore
        "weights": WEIGHTS,  # type: ignore
        "flip_y": FLIP_Y,  # type: ignore
        "class_sep": CLASS_SEP,  # type: ignore
        "hypercube": HYPERCUBE,
        "scale": SCALE,  # type: ignore
        "random_state": RANDOM_SEED,
    }

    log_it(
        "Dataset creating arguments found using variables: "
        + str(synthetic_creation_args)
    )
    X, Y = make_classification(**synthetic_creation_args)  # type: ignore
    log_it("Dataset creation done")
    df = pd.DataFrame(X)
    df["label"] = Y
    return df, synthetic_creation_args
