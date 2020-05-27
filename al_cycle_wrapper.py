from collections import defaultdict
import datetime
import hashlib
import math
import operator
import threading
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#  import np.random.distributions as dists
from json_tricks import dumps
from sklearn.preprocessing import LabelEncoder

from .cluster_strategies import (
    DummyClusterStrategy,
    MostUncertainClusterStrategy,
    RandomClusterStrategy,
    RoundRobinClusterStrategy,
)
from .dataStorage import DataStorage
from .experiment_setup_lib import (
    ExperimentResult,
    calculate_global_score,
    conf_matrix_and_acc,
    get_db,
    get_param_distribution,
    init_logger,
)
from .sampling_strategies import BoundaryPairSampler, RandomSampler, UncertaintySampler


def train_al(
    X_labeled,
    Y_labeled,
    X_unlabeled,
    label_encoder,
    START_SET_SIZE,
    hyper_parameters,
    oracle,
    TEST_FRACTION=None,
    X_test=None,
    Y_test=None,
):
    hyper_parameters["LEN_TRAIN_DATA"] = len(X_labeled)
    dataset_storage = DataStorage(hyper_parameters["RANDOM_SEED"])
    dataset_storage.set_training_data(
        X_labeled,
        Y_labeled,
        X_unlabeled=X_unlabeled,
        START_SET_SIZE=START_SET_SIZE,
        TEST_FRACTION=TEST_FRACTION,
        label_encoder=label_encoder,
        hyper_parameters=hyper_parameters,
        X_test=X_test,
        Y_test=Y_test,
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

    cluster_strategy.set_data_storage(dataset_storage, hyper_parameters["N_JOBS"])

    active_learner_params = {
        "dataset_storage": dataset_storage,
        "cluster_strategy": cluster_strategy,
        "N_JOBS": hyper_parameters["N_JOBS"],
        "RANDOM_SEED": hyper_parameters["RANDOM_SEED"],
        "NR_LEARNING_ITERATIONS": hyper_parameters["NR_LEARNING_ITERATIONS"],
        "NR_QUERIES_PER_ITERATION": hyper_parameters["NR_QUERIES_PER_ITERATION"],
        "oracle": oracle,
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
    #  elif hyper_parameters['sampling'] == 'committee':
    #  active_learner = CommitteeSampler(hyper_parameters['RANDOM_SEED, hyper_parameters.N_JOBS, hyper_parameters.NR_LEARNING_ITERATIONS)
    else:
        ("No Active Learning Strategy specified")

    start = timer()
    trained_active_clf_list, metrics_per_al_cycle, Y_train = active_learner.learn(
        **hyper_parameters
    )
    end = timer()

    return (
        trained_active_clf_list,
        Y_train,
        end - start,
        metrics_per_al_cycle,
        dataset_storage,
        active_learner,
    )


def eval_al(
    X_test,
    Y_test,
    label_encoder,
    trained_active_clf_list,
    fit_time,
    metrics_per_al_cycle,
    dataset_storage,
    active_learner,
    hyper_parameters,
    dataset_name,
    Y_train_al,
    X_train,
    Y_train,
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

    amount_of_all_labels = len(Y_train_al)

    # calculate accuracy for Random Forest only on oracle human expert queries

    active_rf = RandomForestClassifier(random_state=hyper_parameters["RANDOM_SEED"])
    ys_oracle = Y_train_al.loc[Y_train_al.source == "A"]
    active_rf.fit(X_train.iloc[ys_oracle.index], ys_oracle[0])
    acc_test_oracle = accuracy_score(Y_test, active_rf.predict(X_test))

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
    db = get_db(db_name_or_type=hyper_parameters["DB_NAME_OR_TYPE"])

    hyper_parameters["DATASET_NAME"] = dataset_name
    print(hyper_parameters.keys())
    hyper_parameters["cores"] = hyper_parameters["N_JOBS"]
    #  del hyper_parameters["N_JOBS"]

    # lower case all parameters for nice values in database
    hyper_parameters = {k.lower(): v for k, v in hyper_parameters.items()}

    experiment_result = ExperimentResult(
        **hyper_parameters,
        metrics_per_al_cycle=dumps(metrics_per_al_cycle, allow_nan=True),
        fit_time=str(fit_time),
        acc_train=metrics_per_al_cycle["train_acc"][-1],
        acc_test=metrics_per_al_cycle["test_acc"][-1],
        acc_test_oracle=acc_test_oracle,
        fit_score=score,
        param_list_id=param_list_id,
        thread_id=threading.get_ident(),
        end_time=datetime.datetime.now(),
        amount_of_all_labels=amount_of_all_labels,
    )
    experiment_result.save()
    db.close()

    return score


"""
Takes a dataset_path, X, Y, label_encoder and does the following steps:
1. Split data
2. Train AL on the train dataset
3. Evaluate AL on the test dataset
4. Returns fit_score
"""


def train_and_eval_dataset(
    dataset_name,
    X_train,
    X_test,
    Y_train,
    Y_test,
    label_encoder_classes,
    hyper_parameters,
    oracle,
):
    label_encoder = LabelEncoder()
    label_encoder.fit(label_encoder_classes)

    (
        trained_active_clf_list,
        Y_train_al,
        fit_time,
        metrics_per_al_cycle,
        dataStorage,
        active_learner,
    ) = train_al(
        X_train,
        Y_train,
        X_unlabeled=None,
        label_encoder=label_encoder,
        START_SET_SIZE=hyper_parameters["START_SET_SIZE"],
        hyper_parameters=hyper_parameters,
        oracle=oracle,
        TEST_FRACTION=hyper_parameters["TEST_FRACTION"],
        X_test=X_test,
        Y_test=Y_test,
    )

    fit_score = eval_al(
        X_test,
        Y_test,
        label_encoder,
        trained_active_clf_list,
        fit_time,
        metrics_per_al_cycle,
        dataStorage,
        active_learner,
        hyper_parameters,
        dataset_name,
        Y_train_al,
        X_train,
        Y_train,
    )
    return fit_score, Y_train_al
