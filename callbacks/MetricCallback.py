import argparse
import datetime
import os
import random
import sys
import threading
import warnings

import numpy as np

import numpy.random
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .BaseCallback import BaseCallback
from typing import Callable


class MetricCallback(BaseCallback):
    def __init__(self, metric_function: Callable[[ActiveLearner], float]) -> None:
        self.metric_function = metric_function

    def pre_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass

    def post_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        self.values.append(metric_function(active_learner))


def test_f1_metric(active_learner: ActiveLearner) -> List[float]:
    Y_true = active_learner.data_storage.Y[active_learner.data_storage.test_mask]
    Y_pred = active_learner.learner.predict(
        active_learner.data_storage.X[active_learner.data_storage.test_mask]
    )

    return f1_score(Y_true, Y_pred, average="weighted", zero_division=0)


def test_acc_metric(active_learner: ActiveLearner) -> List[float]:
    Y_true = active_learner.data_storage.Y[active_learner.data_storage.test_mask]
    Y_pred = active_learner.learner.predict(
        active_learner.data_storage.X[active_learner.data_storage.test_mask]
    )

    return accuracy_score(Y_true, Y_pred)


#
#  def calculate_post_metrics(self, X_query, Y_query):
#      if len(self.data_storage.test_mask) > 0:
#          # experiment
#          conf_matrix, acc, f1 = conf_matrix_and_acc_and_f1(
#              self.clf,
#              self.data_storage.X[self.data_storage.test_mask],
#              self.data_storage.Y[self.data_storage.test_mask],
#              self.data_storage.label_encoder,
#          )
#      else:
#          conf_matrix, acc, f1 = None, 0, 0
#      self.metrics_per_al_cycle["test_conf_matrix"].append(conf_matrix)
#      self.metrics_per_al_cycle["test_acc"].append(acc)
#      self.metrics_per_al_cycle["test_f1"].append(f1)
#
#      if len(self.data_storage.test_mask) > 0:
#          # experiment
#          conf_matrix, acc, f1 = conf_matrix_and_acc_and_f1(
#              self.clf,
#              self.data_storage.X[self.data_storage.labeled_mask],
#              self.data_storage.Y[self.data_storage.labeled_mask],
#              self.data_storage.label_encoder,
#          )
#      else:
#          conf_matrix, acc, f1 = None, 0, 0
#
#      self.metrics_per_al_cycle["train_conf_matrix"].append(conf_matrix)
#      self.metrics_per_al_cycle["train_acc"].append(acc)
#      self.metrics_per_al_cycle["train_f1"].append(f1)
#
#      if self.data_storage.PLOT_EVOLUTION:
#          self.data_storage.train_unlabeled_Y_predicted = self.clf.predict(
#              self.data_storage.X[self.data_storage.unlabeled_mask]
#          )
#          self.data_storage.train_labeled_Y_predicted = self.clf.predict(
#              self.data_storage.X[self.data_storage.labeled_mask]
#          )
#

#########################


#  def conf_matrix_and_acc_and_f1(clf, X, Y_true, label_encoder):
#      Y_pred = clf.predict(X)
#      conf_matrix = confusion_matrix(Y_true, Y_pred)
#      acc = accuracy_score(Y_true, Y_pred)
#      f1 = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
#      return conf_matrix, acc, f1
#


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
