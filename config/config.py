import argparse
import random
import sys
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.stats import uniform  # type: ignore

from ..logger.logger import init_logger

CliConfigParameters = List[Tuple[List[str], Dict[str, Any]]]


def standard_config(
    additional_parameters: CliConfigParameters = None,
    standard_args: bool = True,
    return_argparse: bool = False,
) -> Union[argparse.Namespace, Tuple[argparse.Namespace, argparse.ArgumentParser]]:
    print("uiuiui")
    parser = argparse.ArgumentParser()
    if standard_args:
        parser.add_argument("--DATASETS_PATH", default="../datasets/")
        parser.add_argument(
            "--CLASSIFIER",
            default="RF",
            help="Supported types: RF, DTree, NB, SVM, Linear",
        )
        parser.add_argument("--N_JOBS", type=int, default=-1)
        parser.add_argument(
            "--RANDOM_SEED", type=int, default=42, help="-1 Enables true Randomness"
        )
        parser.add_argument("--TEST_FRACTION", type=float, default=0.5)
        parser.add_argument("--LOG_FILE", type=str, default="log.txt")

    if additional_parameters is not None:
        for additional_parameter in additional_parameters:
            parser.add_argument(*additional_parameter[0], **additional_parameter[1])

    config: argparse.Namespace = parser.parse_args()
    print("ui")
    print(config)
    if len(sys.argv[:-1]) == 0:
        parser.print_help()
        parser.exit()

    if config.RANDOM_SEED != -1 and config.RANDOM_SEED != -2:
        np.random.seed(config.RANDOM_SEED)
        random.seed(config.RANDOM_SEED)

    init_logger(config.LOG_FILE)
    if return_argparse:
        return config, parser
    else:
        return config


def get_active_config(
    additional_parameters: CliConfigParameters = [],
) -> Union[argparse.Namespace, Tuple[argparse.Namespace, argparse.ArgumentParser]]:
    return standard_config(
        [
            (
                ["--SAMPLING"],
                {
                    "help": "Possible values: uncertainty, random, committe, boundary",
                },
            ),
            (
                ["--DATASET_NAME"],
                {
                    "required": True,
                },
            ),
            (
                ["--CLUSTER"],
                {
                    "default": "dummy",
                    "help": "Possible values: dummy, random, mostUncertain, roundRobin",
                },
            ),
            (["--NR_LEARNING_ITERATIONS"], {"type": int, "default": 150000}),
            (["--BATCH_SIZE"], {"type": int, "default": 150}),
            (["--START_SET_SIZE"], {"type": int, "default": 1}),
            (
                ["--MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS"],
                {"type": float, "default": 0.5},
            ),
            (
                ["--UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"],
                {"type": float, "default": 0.9},
            ),
            (
                ["--UNCERTAINTY_RECOMMENDATION_RATIO"],
                {"type": float, "default": 1 / 100},
            ),
            (
                ["--SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY"],
                {"type": float, "default": 0.9},
            ),
            (
                ["--CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE"],
                {"type": float, "default": 0.7},
            ),
            (
                ["--CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED"],
                {"type": float, "default": 0.9},
            ),
            (["--WITH_UNCERTAINTY_RECOMMENDATION"], {"action": "store_true"}),
            (["--WITH_CLUSTER_RECOMMENDATION"], {"action": "store_true"}),
            (["--WITH_SNUBA_LITE"], {"action": "store_true"}),
            (["--PLOT"], {"action": "store_true"}),
            (["--STOPPING_CRITERIA_UNCERTAINTY"], {"type": float, "default": 0.7}),
            (["--STOPPING_CRITERIA_ACC"], {"type": float, "default": 0.7}),
            (["--STOPPING_CRITERIA_STD"], {"type": float, "default": 0.7}),
            (
                ["--ALLOW_RECOMMENDATIONS_AFTER_STOP"],
                {"action": "store_true", "default": False},
            ),
            (["--OUTPUT_DIRECTORY"], {"default": "tmp/"}),
            (["--HYPER_SEARCH_TYPE"], {"default": "random"}),
            (["--USER_QUERY_BUDGET_LIMIT"], {"type": float, "default": 200}),
            (["--AMOUNT_OF_PEAKED_OBJECTS"], {"type": int, "default": 12}),
            (["--MAX_AMOUNT_OF_WS_PEAKS"], {"type": int, "default": 1}),
            (["--AMOUNT_OF_LEARN_ITERATIONS"], {"type": int, "default": 1}),
            (["--PLOT_EVOLUTION"], {"action": "store_true"}),
            (["--REPRESENTATIVE_FEATURES"], {"action": "store_true"}),
            (["--VARIABLE_DATASET"], {"action": "store_true"}),
            (["--NEW_SYNTHETIC_PARAMS"], {"action": "store_true"}),
            (["--HYPERCUBE"], {"action": "store_true"}),
            (["--AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
            (["--STOP_AFTER_MAXIMUM_ACCURACY_REACHED"], {"action": "store_true"}),
            (["--GENERATE_NOISE"], {"action": "store_true"}),
            (["--STATE_DIFF_PROBAS"], {"action": "store_true"}),
            (["--STATE_ARGSECOND_PROBAS"], {"action": "store_true"}),
            (["--STATE_ARGTHIRD_PROBAS"], {"action": "store_true"}),
            (["--STATE_DISTANCES_LAB"], {"action": "store_true"}),
            (["--STATE_DISTANCES_UNLAB"], {"action": "store_true"}),
            (["--STATE_PREDICTED_CLASS"], {"action": "store_true"}),
            (["--STATE_PREDICTED_UNITY"], {"action": "store_true"}),
            (["--STATE_DISTANCES"], {"action": "store_true"}),
            (["--STATE_UNCERTAINTIES"], {"action": "store_true"}),
            (["--BATCH_MODE"], {"action": "store_true"}),
            (["--DISTANCE_METRIC"], {"default": "euclidean"}),
            (["--STATE_INCLUDE_NR_FEATURES"], {"action": "store_true"}),
            (["--INITIAL_BATCH_SAMPLING_METHOD"], {"default": "furthest"}),
            (["--INITIAL_BATCH_SAMPLING_ARG"], {"type": int, "default": 100}),
            (
                ["--INITIAL_BATCH_SAMPLING_HYBRID_UNCERT"],
                {"type": float, "default": 0.2},
            ),
            (
                ["--INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST"],
                {"type": float, "default": 0.2},
            ),
            (
                ["--INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB"],
                {"type": float, "default": 0.2},
            ),
            (
                ["--INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY"],
                {"type": float, "default": 0.2},
            ),
            *additional_parameters,
        ]
    )


def calculate_unique_config_id(config: argparse.Namespace):
    pass


def get_param_distribution(
    hyper_search_type=None,
    DATASETS_PATH=None,
    CLASSIFIER=None,
    N_JOBS=None,
    RANDOM_SEED=None,
    TEST_FRACTION=None,
    NR_LEARNING_ITERATIONS=None,
    OUTPUT_DIRECTORY=None,
    **kwargs
):
    if hyper_search_type == "random":
        zero_to_one = uniform(loc=0, scale=1)
        half_to_one = uniform(loc=0.5, scale=0.5)
        #  BATCH_SIZE = scipy.stats.randint(1, 151)
        BATCH_SIZE = [10]
        #  START_SET_SIZE = scipy.stats.uniform(loc=0.001, scale=0.1)
        #  START_SET_SIZE = [1, 10, 25, 50, 100]
        START_SET_SIZE = [1]
    else:
        param_size = 50
        #  param_size = 2
        zero_to_one = np.linspace(0, 1, num=param_size * 2 + 1).astype(float)
        half_to_one = np.linspace(0.5, 1, num=param_size + 1).astype(float)
        BATCH_SIZE = [10]  # np.linspace(1, 150, num=param_size + 1).astype(int)
        #  START_SET_SIZE = np.linspace(0.001, 0.1, num=10).astype(float)
        START_SET_SIZE = [1]

    param_distribution = {
        "DATASETS_PATH": [DATASETS_PATH],
        "CLASSIFIER": [CLASSIFIER],
        "N_JOBS": [N_JOBS],
        "RANDOM_SEED": [RANDOM_SEED],
        "TEST_FRACTION": [TEST_FRACTION],
        "SAMPLING": [
            "random",
            "uncertainty_lc",
            "uncertainty_max_margin",
            "uncertainty_entropy",
        ],
        "CLUSTER": [
            "dummy",
            "random",
            "MostUncertain_lc",
            "MostUncertain_max_margin",
            "MostUncertain_entropy"
            #  'dummy',
        ],
        "NR_LEARNING_ITERATIONS": [NR_LEARNING_ITERATIONS],
        #  "NR_LEARNING_ITERATIONS": [1],
        "BATCH_SIZE": BATCH_SIZE,
        "START_SET_SIZE": START_SET_SIZE,
        "STOPPING_CRITERIA_UNCERTAINTY": [1],  # zero_to_one,
        "STOPPING_CRITERIA_STD": [1],  # zero_to_one,
        "STOPPING_CRITERIA_ACC": [1],  # zero_to_one,
        "ALLOW_RECOMMENDATIONS_AFTER_STOP": [True],
        # uncertainty_recommendation_grid = {
        "UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD": np.linspace(
            0.85, 1, num=15 + 1
        ),  # half_to_one,
        "UNCERTAINTY_RECOMMENDATION_RATIO": [
            1 / 100,
            1 / 1000,
            1 / 10000,
            1 / 100000,
            1 / 1000000,
        ],
        # snuba_lite_grid = {
        "SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY": [0],
        #  half_to_one,
        # cluster_recommendation_grid = {
        "CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE": half_to_one,
        "CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED": half_to_one,
        "WITH_UNCERTAINTY_RECOMMENDATION": [True, False],
        "WITH_CLUSTER_RECOMMENDATION": [True, False],
        "WITH_SNUBA_LITE": [False],
        "MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS": half_to_one,
        "OUTPUT_DIRECTORY": [OUTPUT_DIRECTORY],
        "USER_QUERY_BUDGET_LIMIT": [200],
    }

    return param_distribution
