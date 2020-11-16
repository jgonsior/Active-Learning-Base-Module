import argparse
import datetime
import os
import random
import sys
import threading
import warnings

import numpy as np

#  import np.random.distributions as dists
import numpy.random
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_classifier(classifier_name, random_state=None, n_jobs=None):
    if classifier_name == "RF":
        return RandomForestClassifier(
            n_jobs=n_jobs, random_state=random_state, warm_start=False
        )
    elif classifier_name == "SVM":
        return SVC(probability=True, random_state=random_state, warm_start=False)
    elif classifier_name == "MLP":
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        return MLPClassifier(random_state=random_state, verbose=0, warm_start=False)
    elif classifier_name == "LR":
        return LogisticRegression(
            random_state=random_state, verbose=0, warm_start=False, max_iter=10000
        )


# really dirty hack to provide logging as functions instead of objects
def init_logger(logfilepath):
    global logfile_path
    logfile_path = logfilepath


def log_it(message):
    message = (
        "["
        + str(threading.get_ident())
        + "] ["
        + str(datetime.datetime.now())
        + "] "
        + str(message)
    )

    if logfile_path == "console":
        print(message)
    else:
        with open(logfile_path, "a") as f:
            f.write(message + "\n")


def standard_config(
    additional_parameters=None, standard_args=True, return_argparse=False
):
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

    config = parser.parse_args()

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


def get_active_config(additional_parameters=[]):
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
            (["--NR_QUERIES_PER_ITERATION"], {"type": int, "default": 150}),
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
            (["--INITIAL_BATCH_SAMPLING_METHOD"], {"default": "furthest"}),
            (["--INITIAL_BATCH_SAMPLING_ARG"], {"type": int, "default": 100}),
            *additional_parameters,
        ]
    )


def get_best_hyper_params(clf):
    if clf == "RF":
        best_hyper_params = {
            "criterion": "gini",
            "max_depth": 46,
            "max_features": "sqrt",
            "max_leaf_nodes": 47,
            "min_samples_leaf": 16,
            "min_samples_split": 6,
            "n_estimators": 77,
        }
    elif clf == "NB":
        best_hyper_params = {"alpha": 0.7982572902331797}
    elif clf == "SVMPoly":
        best_hyper_params = {}
    elif clf == "SVMRbf":
        best_hyper_params = {
            "C": 1000,
            "cache_size": 10000,
            "gamma": 0.1,
            "kernel": "rbf",
        }

    return best_hyper_params


def conf_matrix_and_acc(clf, X, Y_true, label_encoder):
    Y_pred = clf.predict(X)
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    acc = accuracy_score(Y_true, Y_pred)
    return conf_matrix, acc


class Logger(object):
    # source: https://stackoverflow.com/q/616645
    def __init__(self, filename="log.txt", mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None


def get_single_al_run_stats_table_header():
    return "Iteration: {:>3} {:>6} {:>6} {:>6} {:>6} {:>6} {:>3}".format(
        "I", "L", "U", "Q", "Te", "Tr", "R"
    )


def get_single_al_run_stats_row(
    i, amount_of_labeled, amount_of_unlabeled, metrics_per_al_cycle, index=-1
):
    if amount_of_labeled == None:
        amount_of_labeled = 0
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_labeled += query_length

        amount_of_unlabeled = 2889
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_unlabeled -= query_length

    return "Iteration: {:3,d} {:6,d} {:6,d} {:6,d} {:6.1%} {:6.1%} {:>3}".format(
        i,
        amount_of_labeled,
        amount_of_unlabeled,
        metrics_per_al_cycle["query_length"][index],
        metrics_per_al_cycle["test_acc"][index],
        metrics_per_al_cycle["train_acc"][index],
        metrics_per_al_cycle["source"][index],
    )


def prettify_bytes(bytes):
    """Get human-readable file sizes.
    simplified version of https://pypi.python.org/pypi/hurry.filesize/
    """
    # bytes pretty-printing
    units = [
        (1 << 50, " PB"),
        (1 << 40, " TB"),
        (1 << 30, " GB"),
        (1 << 20, " MB"),
        (1 << 10, " KB"),
        (1, (" byte", " bytes")),
    ]
    for factor, suffix in units:
        if bytes >= factor:
            break
    amount = int(bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix


def calculate_roc_auc(label_encoder, X_test, Y_test, clf):
    #  print(set(Y_test[0].to_numpy()))
    if len(label_encoder.classes_) > 2:
        Y_scores = np.array(clf.predict_proba(X_test))
        #  print(Y_scores)
        Y_test = Y_test.to_numpy().reshape(1, len(Y_scores))[0].tolist()

        return roc_auc_score(
            Y_test,
            Y_scores,
            multi_class="ovo",
            average="macro",
            labels=[i for i in range(len(label_encoder.classes_))],
        )
    else:
        Y_scores = clf.predict_proba(X_test)[:, 1]
        #  print(Y_test.shape)
        Y_test = Y_test.to_numpy().reshape(1, len(Y_scores))[0].tolist()
        return roc_auc_score(Y_test, Y_scores)


# for details see http://www.causality.inf.ethz.ch/activelearning.php?page=evaluation#cont
def calculate_global_score(
    metric_values, amount_of_labels_per_metric_values, amount_of_labels
):
    if len(metric_values) > 1:
        rectangles = []
        triangles = []

        for (
            metric_value,
            amount_of_labels_per_metric_value,
            past_metric_value,
        ) in zip(
            metric_values[1:],
            amount_of_labels_per_metric_values[1:],
            metric_values[:-1],
        ):
            rectangles.append(metric_value * amount_of_labels_per_metric_value)
            triangles.append(
                amount_of_labels_per_metric_value
                * (past_metric_value - metric_value)
                / 2
            )
        square = sum(rectangles) + sum(triangles)
    else:
        square = metric_values[0] * amount_of_labels_per_metric_values[0]

    amax = sum(amount_of_labels_per_metric_values)
    arand = amax * (1 / amount_of_labels)
    global_score = (square - arand) / (amax - arand)

    if global_score > 1:
        print("metric_values: ", metric_values)
        print("#q: ", amount_of_labels_per_metric_values)
        print("rect: ", rectangles)
        print("tria: ", triangles)
        print("ama: ", amax)
        print("ara: ", arand)
        print("squ: ", square)
        print("glob: ", global_score)

    return global_score


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
        zero_to_one = scipy.stats.uniform(loc=0, scale=1)
        half_to_one = scipy.stats.uniform(loc=0.5, scale=0.5)
        #  nr_queries_per_iteration = scipy.stats.randint(1, 151)
        NR_QUERIES_PER_ITERATION = [10]
        #  START_SET_SIZE = scipy.stats.uniform(loc=0.001, scale=0.1)
        #  START_SET_SIZE = [1, 10, 25, 50, 100]
        START_SET_SIZE = [1]
    else:
        param_size = 50
        #  param_size = 2
        zero_to_one = np.linspace(0, 1, num=param_size * 2 + 1).astype(float)
        half_to_one = np.linspace(0.5, 1, num=param_size + 1).astype(float)
        NR_QUERIES_PER_ITERATION = [
            10
        ]  # np.linspace(1, 150, num=param_size + 1).astype(int)
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
        "NR_QUERIES_PER_ITERATION": NR_QUERIES_PER_ITERATION,
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
