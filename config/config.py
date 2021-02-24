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
                {"default": "synthetic"},
            ),
            (
                ["--CLUSTER"],
                {
                    "default": "dummy",
                    "help": "Possible values: dummy, random, mostUncertain, roundRobin",
                },
            ),
            (["--BATCH_SIZE"], {"type": int, "default": 5}),
            (["--PLOT"], {"action": "store_true"}),
            (["--OUTPUT_DIRECTORY"], {"default": "tmp/"}),
            (["--HYPER_SEARCH_TYPE"], {"default": "random"}),
            (["--TOTAL_BUDGET"], {"type": float, "default": 200}),
            (["--AMOUNT_OF_PEAKED_OBJECTS"], {"type": int, "default": 12}),
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
            (
                ["--PRE_SAMPLING_METHOD"],
                {"type": str, "default": "furthest"},
            ),
            (
                ["--PRE_SAMPLING_ARG"],
                {"type": int, "default": 10},
            ),
            *additional_parameters,
        ]
    )
