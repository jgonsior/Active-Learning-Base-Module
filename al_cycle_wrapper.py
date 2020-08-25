import csv
import datetime
import hashlib
import threading
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

#  import np.random.distributions as dists
from json_tricks import dumps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from .dataStorage import DataStorage
from .experiment_setup_lib import get_param_distribution
from .sampling_strategies import (
    BoundaryPairSampler,
    RandomSampler,
    UncertaintySampler,
    OptimalForecastSampler,
    TrainedNNLearner,
)
from .weak_supervision import WeakCert, WeakClust


def train_al(hyper_parameters, oracle, df=None, DATASET_NAME=None, DATASETS_PATH=None):
    data_storage = DataStorage(
        hyper_parameters["RANDOM_SEED"],
        hyper_parameters,
        df=df,
        DATASET_NAME=DATASET_NAME,
        DATASETS_PATH=DATASETS_PATH,
        VARIABLE_INPUT_SIZE=hyper_parameters["VARIABLE_INPUT_SIZE"],
        NEW_SYNTHETIC_PARAMS=hyper_parameters["NEW_SYNTHETIC_PARAMS"],
        HYPERCUBE=hyper_parameters["HYPERCUBE"],
        AMOUNT_OF_FEATURES=hyper_parameters["AMOUNT_OF_FEATURES"],
        #
    )
    hyper_parameters["LEN_TRAIN_DATA"] = len(data_storage.train_unlabeled_Y) + len(
        data_storage.train_labeled_Y
    )

    if hyper_parameters["CLUSTER"] == "dummy":
        cluster_strategy = DummyClusterStrategy()
    elif hyper_parameters["CLUSTER"] == "random":
        cluster_strategy = RandomClusterStrategy()
    elif hyper_parameters["CLUSTER"] == "MostUncertain_lc":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy("least_confident")
    elif hyper_parameters["CLUSTER"] == "MostUncertain_max_margin":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy("max_margin")
    elif hyper_parameters["CLUSTER"] == "MostUncertain_entropy":
        cluster_strategy = MostUncertainClusterStrategy()
        cluster_strategy.set_uncertainty_strategy("entropy")
    elif hyper_parameters["CLUSTER"] == "RoundRobin":
        cluster_strategy = RoundRobinClusterStrategy()

    cluster_strategy.set_data_storage(data_storage, hyper_parameters["N_JOBS"])

    classifier = RandomForestClassifier(
        n_jobs=hyper_parameters["N_JOBS"], random_state=hyper_parameters["RANDOM_SEED"]
    )

    weak_supervision_label_sources = []

    if hyper_parameters["WITH_CLUSTER_RECOMMENDATION"]:
        weak_supervision_label_sources.append(
            WeakClust(
                data_storage,
                MINIMUM_CLUSTER_UNITY_SIZE=hyper_parameters[
                    "CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE"
                ],
                MINIMUM_RATIO_LABELED_UNLABELED=hyper_parameters[
                    "CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED"
                ],
            )
        )

    if hyper_parameters["WITH_UNCERTAINTY_RECOMMENDATION"]:
        weak_supervision_label_sources.append(
            WeakCert(
                data_storage,
                CERTAINTY_THRESHOLD=hyper_parameters[
                    "UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"
                ],
                CERTAINTY_RATIO=hyper_parameters["UNCERTAINTY_RECOMMENDATION_RATIO"],
                clf=classifier,
            )
        )

    active_learner_params = {
        "data_storage": data_storage,
        "cluster_strategy": cluster_strategy,
        "N_JOBS": hyper_parameters["N_JOBS"],
        "RANDOM_SEED": hyper_parameters["RANDOM_SEED"],
        "NR_LEARNING_ITERATIONS": hyper_parameters["NR_LEARNING_ITERATIONS"],
        "NR_QUERIES_PER_ITERATION": hyper_parameters["NR_QUERIES_PER_ITERATION"],
        "oracle": oracle,
        "clf": classifier,
        "weak_supervision_label_sources": weak_supervision_label_sources,
    }

    if hyper_parameters["SAMPLING"] == "random":
        active_learner = RandomSampler(**active_learner_params)
    elif hyper_parameters["SAMPLING"] == "boundary":
        active_learner = BoundaryPairSampler(**active_learner_params)
    elif hyper_parameters["SAMPLING"] == "uncertainty_lc":
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy("least_confident")
    elif hyper_parameters["SAMPLING"] == "uncertainty_max_margin":
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy("max_margin")
    elif hyper_parameters["SAMPLING"] == "uncertainty_entropy":
        active_learner = UncertaintySampler(**active_learner_params)
        active_learner.set_uncertainty_strategy("entropy")
    elif hyper_parameters["SAMPLING"] == "optimal_forecast":
        active_learner = OptimalForecastSampler(**active_learner_params)
        active_learner.set_amount_of_peaked_objects(
            hyper_parameters["AMOUNT_OF_PEAKED_OBJECTS"]
        )
        active_learner.MAX_AMOUNT_OF_WS_PEAKS = hyper_parameters[
            "MAX_AMOUNT_OF_WS_PEAKS"
        ]
    elif hyper_parameters["SAMPLING"] == "trained_nn":
        active_learner = TrainedNNLearner(**active_learner_params)
        active_learner.init_sampling_classifier(
            hyper_parameters["NN_BINARY"],
            hyper_parameters["AMOUNT_OF_RANDOM_QUERY_SETS"],
            hyper_parameters["REPRESENTATIVE_FEATURES"],
        )
        active_learner.MAX_AMOUNT_OF_WS_PEAKS = hyper_parameters[
            "MAX_AMOUNT_OF_WS_PEAKS"
        ]
    #  elif hyper_parameters['sampling'] == 'committee':
    #  active_learner = CommitteeSampler(hyper_parameters['RANDOM_SEED, hyper_parameters.N_JOBS, hyper_parameters.NR_LEARNING_ITERATIONS)
    else:
        ("No Active Learning Strategy specified")

    start = timer()
    trained_active_clf_list, metrics_per_al_cycle = active_learner.learn(
        **hyper_parameters
    )
    end = timer()

    return (
        trained_active_clf_list,
        end - start,
        metrics_per_al_cycle,
        data_storage,
        active_learner,
    )


def eval_al(
    data_storage,
    trained_active_clf_list,
    fit_time,
    metrics_per_al_cycle,
    active_learner,
    hyper_parameters,
):
    hyper_parameters[
        "amount_of_user_asked_queries"
    ] = active_learner.amount_of_user_asked_queries

    # normalize by start_set_size
    percentage_user_asked_queries = (
        1
        - hyper_parameters["amount_of_user_asked_queries"]
        / hyper_parameters["LEN_TRAIN_DATA"]
    )
    test_acc = metrics_per_al_cycle["test_acc"][-1]

    # score is harmonic mean
    score = (
        2
        * percentage_user_asked_queries
        * test_acc
        / (percentage_user_asked_queries + test_acc)
    )

    amount_of_all_labels = len(data_storage.train_labeled_Y)

    # calculate accuracy for Random Forest only on oracle human expert queries

    active_rf = RandomForestClassifier(random_state=hyper_parameters["RANDOM_SEED"])

    Y_train_al = data_storage.train_labeled_Y

    ys_oracle_a = Y_train_al.loc[Y_train_al.source == "A"]
    ys_oracle_g = Y_train_al.loc[Y_train_al.source == "G"]
    ys_oracle = pd.concat([ys_oracle_g, ys_oracle_a])

    active_rf.fit(
        data_storage.train_labeled_X.loc[ys_oracle.index], ys_oracle["label"].to_list()
    )
    acc_test_oracle = accuracy_score(
        data_storage.test_Y, active_rf.predict(data_storage.test_X)
    )

    # save labels
    #  Y_train_al.to_pickle(
    #  "pickles/" + str(len(Y_train_al)) + "_" + param_list_id + ".pickle"
    #  )

    # calculate based on params a unique id which should be the same across all similar cross validation splits
    param_distribution = get_param_distribution(**hyper_parameters)
    unique_params = ""
    for k in param_distribution.keys():
        unique_params += str(hyper_parameters[k])
    param_list_id = hashlib.md5(unique_params.encode("utf-8")).hexdigest()
    #  db = get_db(db_name_or_type=hyper_parameters["DB_NAME_OR_TYPE"])

    #  hyper_parameters["DATASET_NAME"] = DATASET_NAME
    print(hyper_parameters.keys())
    hyper_parameters["cores"] = hyper_parameters["N_JOBS"]
    #  del hyper_parameters["N_JOBS"]

    # lower case all parameters for nice values in database
    hyper_parameters = {k.lower(): v for k, v in hyper_parameters.items()}
    hyper_parameters["fit_time"] = fit_time
    hyper_parameters["metrics_per_al_cycle"] = dumps(
        metrics_per_al_cycle, allow_nan=True
    )
    hyper_parameters["acc_train"] = metrics_per_al_cycle["train_acc"][-1]
    hyper_parameters["acc_test"] = metrics_per_al_cycle["test_acc"][-1]
    hyper_parameters["acc_test_oracle"] = acc_test_oracle
    hyper_parameters["fit_score"] = score
    hyper_parameters["param_list_id"] = param_list_id
    hyper_parameters["thread_id"] = threading.get_ident()
    hyper_parameters["end_time"] = datetime.datetime.now()
    hyper_parameters["amount_of_all_labels"] = amount_of_all_labels

    if hyper_parameters["dataset_name"] == "synthetic":
        hyper_parameters = {**hyper_parameters, **data_storage.synthetic_creation_args}

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

    # save metrics_per_al_cycle in pickle file
    #      metrics_per_al_cycle=dumps(metrics_per_al_cycle, allow_nan=True),
    #      fit_time=str(fit_time),
    #      acc_train=metrics_per_al_cycle["train_acc"][-1],
    #      acc_test=metrics_per_al_cycle["test_acc"][-1],
    #      acc_test_oracle=acc_test_oracle,
    #      fit_score=score,
    #      param_list_id=param_list_id,
    #      thread_id=threading.get_ident(),
    #      end_time=datetime.datetime.now(),
    #      amount_of_all_labels=amount_of_all_labels,

    return score


"""
Takes a dataset_path, X, Y, label_encoder and does the following steps:
1. Split data
2. Train AL on the train dataset
3. Evaluate AL on the test dataset
4. Returns fit_score
"""


def train_and_eval_dataset(
    hyper_parameters, oracle, df=None, DATASET_NAME=None, DATASETS_PATH=None,
):
    (
        trained_active_clf_list,
        fit_time,
        metrics_per_al_cycle,
        data_storage,
        active_learner,
    ) = train_al(
        df=df,
        DATASET_NAME=DATASET_NAME,
        DATASETS_PATH=DATASETS_PATH,
        hyper_parameters=hyper_parameters,
        oracle=oracle,
    )

    fit_score = eval_al(
        data_storage,
        trained_active_clf_list,
        fit_time,
        metrics_per_al_cycle,
        active_learner,
        hyper_parameters,
    )
    return fit_score
