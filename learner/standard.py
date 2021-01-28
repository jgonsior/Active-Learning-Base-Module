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
from sklearn.metrics import accuracy_score, f1_score
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
