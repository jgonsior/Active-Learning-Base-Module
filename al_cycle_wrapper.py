import csv
import datetime
import hashlib
import threading
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from json_tricks import dumps
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score)

from active_learning import callbacks
from active_learning.activeLearner import ActiveLearner
from active_learning.callbacks import (MetricCallback, test_acc_metric,
                                       test_f1_metric)
from active_learning.config.config import get_param_distribution
from active_learning.datasets import (load_alc, load_dwtc, load_synthetic,
                                      load_uci)
from active_learning.dataStorage import DataStorage
from active_learning.learner import Learner, get_classifier
from active_learning.oracles import BaseOracle
from active_learning.sampling_strategies import (BaseSamplingStrategy,
                                                 RandomSampler,
                                                 TrainImitALBatch,
                                                 TrainImitALSingle,
                                                 UncertaintySampler)
from active_learning.stopping_criterias import ALCyclesStoppingCriteria

from .dataStorage import DataStorage


def train_al(
    hyper_parameters: Dict[str, Any],
    oracles: List[BaseOracle],
    data_storage: DataStorage = None,
) -> Tuple[Learner, float, Dict[str, MetricCallback], DataStorage, ActiveLearner]:

    if data_storage is None:
        if hyper_parameters["DATASET_NAME"] == "alc":
            df = load_alc(
                hyper_parameters["DATASETS_PATH"],
                hyper_parameters["DATASET_NAME"],
                hyper_parameters["RANDOM_SEED"],
            )
            data_storage = DataStorage(
                df, TEST_FRACTION=hyper_parameters["TEST_FRACTION"]
            )
        elif hyper_parameters["DATASET_NAME"] == "dwtc":
            df = load_dwtc(
                hyper_parameters["DATASETS_PATH"], hyper_parameters["RANDOM_SEED"]
            )
            data_storage = DataStorage(
                df, TEST_FRACTION=hyper_parameters["TEST_FRACTION"]
            )
        elif hyper_parameters["DATASET_NAME"] == "synthetic":
            df, synthetic_creation_args = load_synthetic(
                hyper_parameters["RANDOM_SEED"],
                hyper_parameters["NEW_SYNTHETIC_PARAMS"],
                hyper_parameters["VARIABLE_DATASET"],
                hyper_parameters["AMOUNT_OF_FEATURES"],
                hyper_parameters["HYPERCUBE"],
                hyper_parameters["GENERATE_NOISE"],
            )
            data_storage = DataStorage(
                df, TEST_FRACTION=hyper_parameters["TEST_FRACTION"]
            )
            hyper_parameters["sythetic_creation_args"] = synthetic_creation_args
        elif hyper_parameters["DATASET_NAME"] == "uci":
            df = load_uci(
                hyper_parameters["DATASETS_PATH"],
                hyper_parameters["DATASEt_NAME"],
                hyper_parameters["RANDOM_SEED"],
            )
            data_storage = DataStorage(
                df, TEST_FRACTION=hyper_parameters["TEST_FRACTION"]
            )
        else:
            print("No dataset found with the given config, exiting")
            exit(-1)

    hyper_parameters["LEN_TRAIN_DATA"] = len(data_storage.unlabeled_mask) + len(
        data_storage.labeled_mask
    )

    sampling_strategy: BaseSamplingStrategy
    if hyper_parameters["SAMPLING"] == "random":
        sampling_strategy = RandomSampler()
    elif hyper_parameters["SAMPLING"] == "uncertainty_lc":
        sampling_strategy = UncertaintySampler("least_confident")
    elif hyper_parameters["SAMPLING"] == "uncertainty_max_margin":
        sampling_strategy = UncertaintySampler("max_margin")
    elif hyper_parameters["SAMPLING"] == "uncertainty_entropy":
        sampling_strategy = UncertaintySampler("entropy")
    elif hyper_parameters["SAMPLING"] == "trained_nn":
        if hyper_parameters["BATCH_MODE"]:
            sampling_strategy = TrainImitALBatch(hyper_parameters["NN_BINARY_PATH"])
        else:
            sampling_strategy = TrainImitALSingle(hyper_parameters["NN_BINARY_PATH"])
    else:
        print("No Active Learning Strategy specified, exiting")
        exit(-1)

    callbacks = {
        "acc_test": MetricCallback(test_acc_metric),
        "f1_test": MetricCallback(test_f1_metric),
    }

    learner = get_classifier(
        hyper_parameters["CLASSIFIER"], random_state=hyper_parameters["RANDOM_SEED"]
    )

    active_learner_params = {
        "sampling_strategy": sampling_strategy,
        "data_storage": data_storage,
        "oracles": oracles,
        "learner": learner,
        "callbacks": callbacks,
        "stopping_criteria": ALCyclesStoppingCriteria(50),
        "BATCH_SIZE": hyper_parameters["PATCH_SIZE"],
    }
    active_learner = ActiveLearner(**active_learner_params)

    start = timer()
    active_learner.al_cycle()
    end = timer()

    return learner, end - start, callbacks, data_storage, active_learner


def eval_al(
    data_storage: DataStorage,
    fit_time: float,
    callbacks: Dict[str, MetricCallback],
    hyper_parameters: Dict[str, Any],
):

    # normalize by start_set_size
    percentage_user_asked_queries = (
        1
        - hyper_parameters["amount_of_user_asked_queries"]
        / hyper_parameters["LEN_TRAIN_DATA"]
    )
    test_acc = callbacks["acc_test"].values[-1]

    # score is harmonic mean
    score = (
        2
        * percentage_user_asked_queries
        * test_acc
        / (percentage_user_asked_queries + test_acc)
    )

    amount_of_all_labels = len(data_storage.labeled_mask)

    # calculate accuracy for Random Forest only on oracle human expert queries
    active_rf = get_classifier(hyper_parameters["CLASSIFIER"])

    active_rf.fit(
        data_storage.X[data_storage.labeled_mask],
        data_storage.Y[data_storage.labeled_mask]
        #  data_storage.train_labeled_X.loc[ys_oracle.index], ys_oracle["label"].to_list()
    )

    y_pred = active_rf.predict(data_storage.X[data_storage.test_mask])
    acc_test_oracle = accuracy_score(data_storage.Y[data_storage.test_mask], y_pred)

    f1_test_oracle = f1_score(
        data_storage.Y[data_storage.test_mask],
        y_pred,
        average="weighted",
        zero_division=0,
    )
    precision_test_oracle = precision_score(
        data_storage.Y[data_storage.test_mask],
        y_pred,
        average="weighted",
        zero_division=0,
    )
    recall_test_oracle = recall_score(
        data_storage.Y[data_storage.test_mask],
        y_pred,
        average="weighted",
        zero_division=0,
    )

    acc_auc = (
        auc(
            [i for i in range(0, len(callbacks["acc_test"].values))],
            callbacks["acc_test"].values,
        )
        / (len(callbacks["acc_test"].values) - 1)
    )
    hyper_parameters["acc_auc"] = acc_auc
    f1_auc = (
        auc(
            [i for i in range(0, len(callbacks["f1_test"].values))],
            callbacks["f1_test"].values,
        )
        / (len(callbacks["f1_test"].values) - 1)
    )

    # calculate based on params a unique id which should be the same across all similar cross validation splits
    param_distribution = get_param_distribution(**hyper_parameters)
    unique_params = ""
    for k in param_distribution.keys():
        unique_params += str(hyper_parameters[k])
    param_list_id = hashlib.md5(unique_params.encode("utf-8")).hexdigest()
    #  db = get_db(db_name_or_type=hyper_parameters["DB_NAME_OR_TYPE"])

    #  hyper_parameters["DATASET_NAME"] = DATASET_NAME
    #  print(hyper_parameters.keys())
    hyper_parameters["cores"] = hyper_parameters["N_JOBS"]
    #  del hyper_parameters["N_JOBS"]

    # lower case all parameters for nice values in database
    hyper_parameters = {k.lower(): v for k, v in hyper_parameters.items()}
    hyper_parameters["fit_time"] = fit_time
    hyper_parameters["metrics_per_al_cycle"] = dumps(str(callbacks), allow_nan=True)
    hyper_parameters["acc_test"] = callbacks["acc_test"].values[-1]
    hyper_parameters["acc_test_oracle"] = acc_test_oracle
    hyper_parameters["f1_test_oracle"] = f1_test_oracle
    hyper_parameters["precision_test_oracle"] = precision_test_oracle
    hyper_parameters["recall_test_oracle"] = recall_test_oracle
    hyper_parameters["acc_auc"] = acc_auc
    hyper_parameters["f1_auc"] = f1_auc
    hyper_parameters["fit_score"] = score
    hyper_parameters["param_list_id"] = param_list_id
    hyper_parameters["thread_id"] = threading.get_ident()
    hyper_parameters["end_time"] = datetime.datetime.now()
    hyper_parameters["amount_of_all_labels"] = amount_of_all_labels

    if hyper_parameters["dataset_name"] == "synthetic":
        hyper_parameters = {
            **hyper_parameters,
            **hyper_parameters["sythetic_creation_args"],
        }

    # save hyper parameter results in csv file
    if hyper_parameters["output_directory"].endswith(".csv"):
        output_hyper_parameter_file = Path(hyper_parameters["output_directory"])
    else:
        output_hyper_parameter_file = Path(
            hyper_parameters["output_directory"] + "/dataset_creation.csv"
        )

    if not output_hyper_parameter_file.is_file():
        output_hyper_parameter_file.touch()
        with output_hyper_parameter_file.open("a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=hyper_parameters.keys())
            csv_writer.writeheader()

    with output_hyper_parameter_file.open("a") as f:
        csv_writer = csv.DictWriter(f, fieldnames=hyper_parameters.keys())
        csv_writer.writerow(hyper_parameters)

    return score


"""
Takes a dataset_path, X, Y, label_encoder and does the following steps:
1. Split data
2. Train AL on the train dataset
3. Evaluate AL on the test dataset
4. Returns fit_score
"""


def train_and_eval_dataset(
    hyper_parameters: Dict[str, Any],
    oracles: List[BaseOracle],
) -> float:
    (_, fit_time, callbacks, data_storage, _,) = train_al(
        hyper_parameters=hyper_parameters,
        oracles=oracles,
        data_storage=None,
    )

    fit_score = eval_al(
        data_storage,
        fit_time,
        callbacks,
        hyper_parameters,
    )
    return fit_score
