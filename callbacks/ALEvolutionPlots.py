import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from numba import jit
from scipy.sparse import lil_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from .experiment_setup_lib import log_it

# @todo implement
class ALEvolutionPlots(BaseCallback):
    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def pre_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass

    @abc.abstractmethod
    def post_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass


if self.PLOT_EVOLUTION:
    self.possible_samples_indices = []
    self.train_labeled_Y_predicted = []
    self.train_unlabeled_Y_predicted = []
    self.i = 0
    self.deleted = False


if self.PLOT_EVOLUTION and source != "P":
    if len(self.train_labeled_Y_predicted) == 0:
        self.i += 1
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.set_size_inches(18.5, 10.5)
        main_hex_colors = ["#3182BD", "#E6550D", "#31A354", "#756BB1"]
        color_list = []
        color_list_r = []
        for i in range(0, self.synthetic_creation_args["n_classes"]):
            color_list += sns.light_palette(main_hex_colors[i]).as_hex()

        cmap = ListedColormap(color_list)
        cmap_bright = ListedColormap(
            main_hex_colors[: self.synthetic_creation_args["n_classes"]]
        )

        x = pd.concat(
            [self.train_labeled_X.iloc[:, 0], self.train_unlabeled_X.iloc[:, 0]]
        )

        y = pd.concat(
            [self.train_labeled_X.iloc[:, 1], self.train_unlabeled_X.iloc[:, 1]]
        )

        c = pd.concat(
            [self.train_labeled_Y.iloc[:, 0], self.train_unlabeled_Y.iloc[:, 0]]
        ).to_numpy()

        c2 = np.concatenate(
            [self.train_labeled_Y_predicted, self.train_unlabeled_Y_predicted]
        )

        areas = []

        for s in self.train_labeled_Y["source"]:
            if s == "G":
                areas.append(1000)
            else:
                areas.append(100)

        for ix, _ in self.train_unlabeled_Y.iterrows():

            if ix == query_indices[0]:
                areas.append(1000)
            else:
                areas.append(10)

        ax1.scatter(
            x=x,
            y=y,
            c=c,
            cmap=cmap_bright,
            alpha=0.5,
            s=areas,
        )
        ax2.scatter(
            x=x,
            y=y,
            c=c2,
            cmap=cmap_bright,
            s=areas,
        )

        # create decision boundary mesh grid
        h = 0.02
        xx, yy = np.meshgrid(np.arange(0, 1.02, h), np.arange(0, 1.02, h))
        db = []

        decision_boundary = self.clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        #  log_it(decision_boundary)

        db = np.argmax(decision_boundary, axis=1) + np.amax(decision_boundary, axis=1)
        db = db.reshape(xx.shape)

        cs = ax2.contourf(
            xx,
            yy,
            db,
            levels=np.arange(0, self.synthetic_creation_args["n_classes"] + 0.1, 0.1),
            cmap=cmap,
            alpha=0.8,
        )

        for peaked_sample in self.possible_samples_indices:
            ax1.add_artist(
                plt.Circle(
                    (self.train_unlabeled_X.loc[peaked_sample]),
                    0.01,
                    fill=False,
                    color="red",
                )
            )
            ax2.add_artist(
                plt.Circle(
                    (self.train_unlabeled_X.loc[peaked_sample]),
                    0.01,
                    fill=False,
                    color="red",
                )
            )

        for current_sample in self.train_unlabeled_X.loc[query_indices].to_numpy():
            ax1.add_artist(
                plt.Circle(
                    (current_sample),
                    0.1,
                    fill=False,
                    color="green",
                )
            )

            ax2.add_artist(
                plt.Circle(
                    (current_sample),
                    0.1,
                    fill=False,
                    color="green",
                )
            )
        cbar = fig.colorbar(cs)

        # highlight misclassified
        train_X = pd.concat([self.train_unlabeled_X, self.train_labeled_X])
        Y_pred = self.clf.predict(train_X)
        Y_true = np.array(
            self.train_unlabeled_Y["label"].to_list()
            + self.train_labeled_Y["label"].to_list()
        )

        misclassified_mask = Y_pred != Y_true
        misclassified_X = train_X[misclassified_mask]
        ax2.scatter(
            x=misclassified_X[0],
            y=misclassified_X[1],
            c="red",
            #  cmap=cmap_bright,
            s=40,
        )

        plt.title("{}: {:.2%} {}".format(self.i, self.test_accuracy, self.deleted))
        plt.savefig(
            self.hyper_parameters["OUTPUT_DIRECTORY"]
            + "/"
            + str(self.RANDOM_SEED)
            + "_"
            + str(self.i)
            + ".png"
        )
        plt.clf()
        self.i += 1
