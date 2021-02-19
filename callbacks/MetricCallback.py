from typing import TYPE_CHECKING, Callable, List

import numpy as np
import numpy.random
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner

from .BaseCallback import BaseCallback


class MetricCallback(BaseCallback):
    def __init__(self, metric_function: Callable[['ActiveLearner'], List[float]]) -> None:
        self.metric_function = metric_function

    def pre_learning_cycle_hook(self, active_learner: 'ActiveLearner') -> None:
        pass

    def post_learning_cycle_hook(self, active_learner: 'ActiveLearner') -> None:
        self.values.append(self.metric_function(active_learner))


def test_f1_metric(active_learner: 'ActiveLearner') -> List[float]:
    Y_true = active_learner.data_storage.Y[active_learner.data_storage.test_mask]
    Y_pred = active_learner.learner.predict(
        active_learner.data_storage.X[active_learner.data_storage.test_mask]
    )

    return f1_score(Y_true, Y_pred, average="weighted", zero_division=0)  # type: ignore


def test_acc_metric(active_learner: 'ActiveLearner') -> List[float]:
    Y_true = active_learner.data_storage.Y[active_learner.data_storage.test_mask]
    Y_pred = active_learner.learner.predict(
        active_learner.data_storage.X[active_learner.data_storage.test_mask]
    )

    return accuracy_score(Y_true, Y_pred)  # type: ignore


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
