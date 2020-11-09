import copy
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import lil_matrix
import math
import seaborn as sns
import random
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from .experiment_setup_lib import log_it


# refactoring: dataset wird ein pandas dataframe:
# id, feature_columns, label (-1 heißt gibt's noch nie, kann auch weak sein), true_label, dataset (train, test, val?), label_source


class DataStorage:
    def __init__(
        self,
        RANDOM_SEED,
        hyper_parameters,
        df=None,
        TEST_FRACTION=0.5,
        DATASET_NAME=None,
        DATASETS_PATH=None,
        PLOT_EVOLUTION=False,
        INITIAL_BATCH_SAMPLING_METHOD="",
        **kwargs
    ):
        if RANDOM_SEED != -1:
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)
            self.RANDOM_SEED = RANDOM_SEED
        self.PLOT_EVOLUTION = PLOT_EVOLUTION

        if PLOT_EVOLUTION:
            self.possible_samples_indices = []
            self.train_labeled_Y_predicted = []
            self.train_unlabeled_Y_predicted = []
            self.i = 0
            self.deleted = False

        self.INITIAL_BATCH_SAMPLING_METHOD = INITIAL_BATCH_SAMPLING_METHOD

        if df is None:
            log_it("Loading " + DATASET_NAME)
            if DATASET_NAME == "dwtc":
                X, Y = self._load_dwtc(DATASETS_PATH)
            elif DATASET_NAME == "synthetic":
                X, Y = self._load_synthetic(RANDOM_SEED=RANDOM_SEED, **kwargs)
            else:
                X, Y = self._load_uci(DATASET_NAME, DATASETS_PATH)
                #  df = self._load_alc(DATASET_NAME, DATASETS_PATH)

        self.hyper_parameters = hyper_parameters

        self.label_encoder = LabelEncoder()

        # ignore nan as labels
        Y = self.label_encoder.fit_transform(Y[~np.isnan(Y)])

        # feature normalization
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

        # scale back to [0,1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        self.label_source = np.full(len(Y), "N", dtype=str)
        self.X = X
        if self.INITIAL_BATCH_SAMPLING_METHOD == "graph_density":
            # compute k-nearest neighbors grap
            self.compute_graph_density()

        # check if we are in an experiment setting or are dealing with real, unlabeled data
        if sum(np.isnan(Y) > 0):
            self.unlabeled_mask = np.argwhere(np.isnan(Y))
            self.labeled_mask = np.argwhere(~np.isnan(Y))
            self.label_source[self.labeled_mask] = ["G" for _ in self.labeled_mask]
            self.Y = np.full(len(self.labeled_mask), np.nan, dtype=np.int64)
            self.Y[self.labeled_mask] = Y

            # create test split out of labeled data
            self.test_mask = self.labeled_mask[
                0 : math.floor(len(self.labeled_mask) * TEST_FRACTION)
            ]
            self.labeled_mask = self.labeled_mask[
                math.floor(len(self.labeled_mask) * TEST_FRACTION) :
            ]

        else:
            # split into test, train_labeled, train_unlabeled
            self.test_mask = np.arange(math.floor(len(Y) * TEST_FRACTION), len(Y))
            # experiment setting apparently
            self.unlabeled_mask = np.arange(0, math.floor(len(Y) * TEST_FRACTION))
            self.labeled_mask = np.empty(0, dtype=np.int64)
            self.Y = Y

            """ 
            1. get start_set from X_labeled
            2. if X_unlabeled is None : experiment!
                2.1 if X_test: rest von X_labeled wird X_train_unlabeled
                2.2 if X_test is none: split rest von X_labeled in X_train_unlabeled und X_test
               else (kein experiment):
               X_unlabeled wird X_unlabeled, rest von X_labeled wird X_train_unlabeled_
            
            """
            # separate X_labeled into start_set and labeled _rest
            # check if the minimum amount of labeled data is present in the start set size
            labels_not_in_start_set = set(range(0, len(self.label_encoder.classes_)))
            all_label_in_start_set = False

            if not all_label_in_start_set:
                #  if len(self.train_labeled_data) == 0:
                #      print("Please specify at least one labeled example of each class")
                #      exit(-1)

                # move more data here from the classes not present
                for label in labels_not_in_start_set:
                    # select a random sample of this labelwhich is NOT yet labeled
                    random_index = np.where(self.Y[self.unlabeled_mask] == label)[0][0]

                    # the random_index before is an index on Y[unlabeled_mask], and therefore NOT the same as an index on purely Y
                    # therefore it needs to be converted first
                    random_index = self.unlabeled_mask[random_index]

                    self._label_samples_without_clusters([random_index], label, "G")

        len_train_labeled = len(self.labeled_mask)
        len_train_unlabeled = len(self.unlabeled_mask)
        #  len_test = len(self.X_test)

        len_total = len_train_unlabeled + len_train_labeled  # + len_test

        log_it(
            "size of train  labeled set: %i = %1.2f"
            % (len_train_labeled, len_train_labeled / len_total)
        )
        log_it(
            "size of train unlabeled set: %i = %1.2f"
            % (len_train_unlabeled, len_train_unlabeled / len_total)
        )

        log_it("Loaded " + str(DATASET_NAME))

    def _load_uci(self, DATASET_NAME, DATASETS_PATH):
        df = pd.read_csv(DATASETS_PATH + "uci_cleaned/" + DATASET_NAME + ".csv")

        # shuffle df
        df = df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        self.synthetic_creation_args = {}
        self.synthetic_creation_args["n_classes"] = len(df["LABEL"].unique())

        Y = df["LABEL"].to_numpy()
        le = LabelEncoder()
        Y = le.fit_transform(Y)

        return df.loc[:, df.columns != "LABEL"].to_numpy(), Y

    def _load_dwtc(self, DATASETS_PATH):
        df = pd.read_csv(DATASETS_PATH + "dwtc/aft.csv")

        # shuffle df
        df = df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        self.synthetic_creation_args = {}
        self.synthetic_creation_args["n_classes"] = len(df["CLASS"].unique())

        Y = df["CLASS"].to_numpy()
        le = LabelEncoder()
        Y = le.fit_transform(Y)

        return df.loc[:, df.columns != "CLASS"].to_numpy(), Y

    def _load_synthetic(self, RANDOM_SEED, **kwargs):
        no_valid_synthetic_arguments_found = True
        while no_valid_synthetic_arguments_found:
            log_it(kwargs)
            if not kwargs["NEW_SYNTHETIC_PARAMS"]:
                if kwargs["VARIABLE_DATASET"]:
                    N_SAMPLES = random.randint(500, 20000)
                else:
                    N_SAMPLES = random.randint(100, 5000)

                if kwargs["AMOUNT_OF_FEATURES"] > 0:
                    N_FEATURES = kwargs["AMOUNT_OF_FEATURES"]
                else:
                    N_FEATURES = random.randint(10, 100)

                N_INFORMATIVE, N_REDUNDANT, N_REPEATED = [
                    int(N_FEATURES * i)
                    for i in np.random.dirichlet(np.ones(3), size=1).tolist()[0]
                ]

                N_CLASSES = random.randint(2, 10)
                N_CLUSTERS_PER_CLASS = random.randint(
                    1, min(max(1, int(2 ** N_INFORMATIVE / N_CLASSES)), 10)
                )

                if N_CLASSES * N_CLUSTERS_PER_CLASS > 2 ** N_INFORMATIVE:
                    continue
                no_valid_synthetic_arguments_found = False

                WEIGHTS = np.random.dirichlet(np.ones(N_CLASSES), size=1).tolist()[
                    0
                ]  # list of weights, len(WEIGHTS) = N_CLASSES, sum(WEIGHTS)=1

                if kwargs["GENERATE_NOISE"]:
                    FLIP_Y = (
                        np.random.pareto(2.0) + 1
                    ) * 0.01  # amount of noise, larger values make it harder
                else:
                    FLIP_Y = 0

                CLASS_SEP = random.uniform(
                    0, 10
                )  # larger values spread out the clusters and make it easier
                HYPERCUBE = kwargs["HYPERCUBE"]  # if false random polytope
                SCALE = 0.01  # features should be between 0 and 1 now
            else:
                if kwargs["VARIABLE_DATASET"]:
                    N_SAMPLES = random.randint(500, 20000)
                else:
                    N_SAMPLES = random.randint(100, 5000)
                    #  N_SAMPLES = 1000
                if kwargs["AMOUNT_OF_FEATURES"] > 0:
                    N_FEATURES = kwargs["AMOUNT_OF_FEATURES"]
                else:
                    N_FEATURES = random.randint(2, 10)
                N_REDUNDANT = N_REPEATED = 0
                N_INFORMATIVE = N_FEATURES

                N_CLASSES = random.randint(2, min(10, 2 ** N_INFORMATIVE))
                N_CLUSTERS_PER_CLASS = random.randint(
                    1, int(2 ** N_INFORMATIVE / N_CLASSES)
                )

                if N_CLASSES * N_CLUSTERS_PER_CLASS > 2 ** N_INFORMATIVE:
                    print("ui")
                    continue
                no_valid_synthetic_arguments_found = False

                WEIGHTS = np.random.dirichlet(np.ones(N_CLASSES), size=1).tolist()[
                    0
                ]  # list of weights, len(WEIGHTS) = N_CLASSES, sum(WEIGHTS)=1

                if kwargs["GENERATE_NOISE"]:
                    FLIP_Y = (
                        np.random.pareto(2.0) + 1
                    ) * 0.01  # amount of noise, larger values make it harder
                else:
                    FLIP_Y = 0

                CLASS_SEP = random.uniform(
                    0, 10
                )  # larger values spread out the clusters and make it easier

                HYPERCUBE = kwargs[
                    "HYPERCUBE"
                ]  # if false a random polytope is selected instead
                SCALE = 0.01  # features should be between 0 and 1 now

            synthetic_creation_args = {
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "n_informative": N_INFORMATIVE,
                "n_redundant": N_REDUNDANT,
                "n_repeated": N_REPEATED,
                "n_classes": N_CLASSES,
                "n_clusters_per_class": N_CLUSTERS_PER_CLASS,
                "weights": WEIGHTS,
                "flip_y": FLIP_Y,
                "class_sep": CLASS_SEP,
                "hypercube": HYPERCUBE,
                "scale": SCALE,
                "random_state": RANDOM_SEED,
            }
            self.synthetic_creation_args = synthetic_creation_args

        log_it(synthetic_creation_args)
        X, Y = make_classification(**synthetic_creation_args)

        return X, Y

    def _load_alc(self, DATASET_NAME, DATASETS_PATH):
        df = pd.read_csv(
            DATASETS_PATH + "/al_challenge/" + DATASET_NAME + ".data",
            header=None,
            sep=" ",
        )
        # feature_columns fehlt

        # shuffle df
        df = df.sample(frac=1, random_state=RANDOM_SEED)

        df = df.replace([np.inf, -np.inf], -1)
        df = df.fillna(0)

        labels = pd.read_csv(
            DATASETS_PATH + "/al_challenge/" + DATASET_NAME + ".label", header=None
        )

        labels = labels.replace([-1], "A")
        labels = labels.replace([1], "B")

        return df.to_numpy(), labels[0].to_numpy()

    # adapted from https://github.com/google/active-learning/blob/master/sampling_methods/graph_density.py#L47-L72
    # original idea: https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf
    def compute_graph_density(self, n_neighbor=10):
        shape = np.shape(self.X)
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))

        gamma = 1.0 / shape[1]

        # kneighbors graph is constructed using k=10
        connect = kneighbors_graph(
            flat_X, n_neighbor, metric="manhattan"
        )  # , n_jobs=self.N_JOBS)

        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()

        inds = zip(neighbors[0], neighbors[1])

        # changes as in connect[i, j] = new_weight are much faster for lil_matrix
        connect_lil = lil_matrix(connect)

        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]

            # das hier auf einmal berechnen?!
            distance = pairwise_distances(
                flat_X[[i]], flat_X[[j]], metric="manhattan"  # , n_jobs=self.N_JOBS
            )

            distance = distance[0, 0]

            # gaussian kernel
            weight = np.exp(-distance * gamma)
            connect_lil[i, j] = weight
            connect_lil[j, i] = weight

        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.

        self.graph_density = np.zeros(shape[0])
        for i in np.arange(shape[0]):
            self.graph_density[i] = (
                connect_lil[i, :].sum() / (connect_lil[i, :] > 0).sum()
            )
        self.connect_lil = connect_lil
        #  self.starting_density = copy.deepcopy(self.graph_density)

    def _label_samples_without_clusters(self, query_indices, Y_query, source):
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

                decision_boundary = self.clf.predict_proba(
                    np.c_[xx.ravel(), yy.ravel()]
                )
                #  log_it(decision_boundary)

                db = np.argmax(decision_boundary, axis=1) + np.amax(
                    decision_boundary, axis=1
                )
                db = db.reshape(xx.shape)

                cs = ax2.contourf(
                    xx,
                    yy,
                    db,
                    levels=np.arange(
                        0, self.synthetic_creation_args["n_classes"] + 0.1, 0.1
                    ),
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

                for current_sample in self.train_unlabeled_X.loc[
                    query_indices
                ].to_numpy():
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

                plt.title(
                    "{}: {:.2%} {}".format(self.i, self.test_accuracy, self.deleted)
                )
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
        # remove before performance measurements -> only a development safety measure
        assert len(np.intersect1d(query_indices, self.labeled_mask)) == 0
        print(query_indices)
        print(self.test_mask)
        print(self.unlabeled_mask)
        print(self.labeled_mask)
        print(np.intersect1d(query_indices, self.test_mask))
        assert len(np.intersect1d(query_indices, self.test_mask)) == 0
        assert len(np.intersect1d(query_indices, self.unlabeled_mask)) == len(
            query_indices
        )

        # -> select samples as a batch and update graph denisty in one run for it completely, or do it on the fly while collecting batch?!
        # --> compare both in opti_setting!
        #  -> in RALF im Detail nachlesen, wie genau die Updates vom density graph aussehen, wann kommt -1, etc.
        if self.INITIAL_BATCH_SAMPLING_METHOD == "graph_density":
            #  print(query_indices)
            #  print(self.connect_lil[query_indices])
            #  print(self.connect_lil[query_indices, :])
            #  print(self.connect_lil[query_indices, :] > 0)
            #  print((self.connect_lil[query_indices, :] > 0).nonzero())
            #  print((self.connect_lil[query_indices, :] > 0).nonzero()[1])
            #  exit(-1)
            neighbors = (self.connect_lil[query_indices, :] > 0).nonzero()[1]
            self.graph_density[neighbors] = (
                self.graph_density[neighbors] - self.graph_density[query_indices]
            )
            print(neighbors)

        self.labeled_mask = np.append(self.labeled_mask, query_indices, axis=0)

        for element in query_indices:
            self.unlabeled_mask = self.unlabeled_mask[self.unlabeled_mask != element]

        self.label_source[query_indices] = source
        # is not working with initial labels, after that it works, but isn't needed
        #  self.Y[query_indices] = Y_query
        # remove before performance measurements -> only a development safety measure
        assert len(np.intersect1d(query_indices, self.unlabeled_mask)) == 0
        assert len(np.intersect1d(query_indices, self.test_mask)) == 0
        assert len(np.intersect1d(query_indices, self.labeled_mask)) == len(
            query_indices
        )

    def label_samples(self, query_indices, Y_query, source):
        # remove from train_unlabeled_data and add to train_labeled_data
        self._label_samples_without_clusters(query_indices, Y_query, source)

        # removed clustering code
        # remove indices from all clusters in unlabeled and add to labeled
        #  for cluster_id in self.train_unlabeled_cluster_indices.keys():
        #      list_to_be_removed_and_appended = []
        #      for indice in query_indices:
        #          if indice in self.train_unlabeled_cluster_indices[cluster_id]:
        #              list_to_be_removed_and_appended.append(indice)
        #
        #      # don't change a list you're iterating over!
        #      for indice in list_to_be_removed_and_appended:
        #          self.train_unlabeled_cluster_indices[cluster_id].remove(indice)
        #          self.train_labeled_cluster_indices[cluster_id].append(indice)
        #
        #  # remove possible empty clusters
        #  self.train_unlabeled_cluster_indices = {
        #      k: v for k, v in self.train_unlabeled_cluster_indices.items() if len(v) != 0
        #  }

    def get_true_label(self, query_indice):
        return self.Y[query_indice]
