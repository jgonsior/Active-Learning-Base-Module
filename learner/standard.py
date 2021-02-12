import argparse
import datetime
import os
import random
import sys
import threading
import warnings
from typing import Any, Dict, Union

import numpy as np
#  import np.random.distributions as dists
import numpy.random
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

Learner = Union[
    MLPClassifier,
    SVC,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GaussianNB,
    LogisticRegression,
]


def get_classifier(
    classifier_name: str, random_state: Union[int, None] = None, n_jobs: int = None
) -> Learner:
    if classifier_name == "RF":
        return RandomForestClassifier(
            n_jobs=n_jobs, random_state=random_state, warm_start=False
        )
    elif classifier_name == "SVM":
        return SVC(probability=True, random_state=random_state)
    elif classifier_name == "MLP":
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        return MLPClassifier(random_state=random_state, verbose=0, warm_start=False)
    elif classifier_name == "LR":
        return LogisticRegression(
            random_state=random_state, verbose=0, warm_start=False, max_iter=10000
        )
    else:
        print("No suitable classifier found for config option, exiting")
        exit(1)


def get_best_hyper_params(clf: str) -> Dict[str, Any]:
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
    else:
        print("No suitable classifier found for config option, exiting")
        exit(-1)

    return best_hyper_params
