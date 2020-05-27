import abc
import collections
import itertools
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .experiment_setup_lib import (
    conf_matrix_and_acc,
    get_single_al_run_stats_row,
    get_single_al_run_stats_table_header,
    log_it,
)


class ActiveLearner:
    def __init__(
        self,
        RANDOM_SEED,
        dataset_storage,
        cluster_strategy,
        N_JOBS,
        NR_LEARNING_ITERATIONS,
        NR_QUERIES_PER_ITERATION,
        oracle,
    ):
        if RANDOM_SEED != -1:
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)

            self.best_hyper_parameters = {"random_state": RANDOM_SEED, "n_jobs": N_JOBS}
        else:
            self.best_hyper_parameters = {"n_jobs": N_JOBS}

        self.data_storage = dataset_storage
        self.NR_LEARNING_ITERATIONS = NR_LEARNING_ITERATIONS
        self.nr_queries_per_iteration = NR_QUERIES_PER_ITERATION

        # it's a list because of committee (in all other cases it's just one CLASSIFIER)
        self.clf = RandomForestClassifier(**self.best_hyper_parameters)

        self.metrics_per_al_cycle = {
            "test_acc": [],
            "test_conf_matrix": [],
            "train_acc": [],
            "train_conf_matrix": [],
            "query_length": [],
            "recommendation": [],
            "labels_indices": [],
        }

        self.cluster_strategy = cluster_strategy
        self.amount_of_user_asked_queries = 0
        self.oracle = oracle

    @abc.abstractmethod
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        pass

    def fit_clf(self):
        self.clf.fit(
            self.data_storage.X_train_labeled,
            self.data_storage.Y_train_labeled[0],
            sample_weight=compute_sample_weight(
                "balanced", self.data_storage.Y_train_labeled[0]
            ),
        )

    def calculate_pre_metrics(self, X_query, Y_query):
        pass

    def calculate_post_metrics(self, X_query, Y_query):

        conf_matrix, acc = conf_matrix_and_acc(
            self.clf,
            self.data_storage.X_test,
            self.data_storage.Y_test[0],
            self.data_storage.label_encoder,
        )

        self.metrics_per_al_cycle["test_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["test_acc"].append(acc)

        if self.data_storage.Y_train_unlabeled.shape[0] > 0:
            # experiment
            conf_matrix, acc = conf_matrix_and_acc(
                self.clf,
                self.data_storage.X_train_labeled,
                self.data_storage.Y_train_labeled[0],
                self.data_storage.label_encoder,
            )
        else:
            conf_matrix, acc = None, 0

        self.metrics_per_al_cycle["train_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["train_acc"].append(acc)

    def cluster_recommendation(
        self, MINIMUM_CLUSTER_UNITY_SIZE, MINIMUM_RATIO_LABELED_UNLABELED
    ):
        certain_X = recommended_labels = certain_indices = None
        cluster_found = False

        # check if the most prominent label for one cluster can be propagated over to the rest of it's cluster
        for (
            cluster_id,
            cluster_indices,
        ) in self.data_storage.X_train_labeled_cluster_indices.items():
            if (
                cluster_id
                not in self.data_storage.X_train_unlabeled_cluster_indices.keys()
            ):
                continue
            if (
                len(cluster_indices)
                / len(self.data_storage.X_train_unlabeled_cluster_indices[cluster_id])
                > MINIMUM_CLUSTER_UNITY_SIZE
            ):
                frequencies = collections.Counter(
                    self.data_storage.Y_train_labeled.loc[cluster_indices][0].tolist()
                )

                if (
                    frequencies.most_common(1)[0][1]
                    > len(cluster_indices) * MINIMUM_RATIO_LABELED_UNLABELED
                ):
                    certain_indices = self.data_storage.X_train_unlabeled_cluster_indices[
                        cluster_id
                    ]

                    certain_X = self.data_storage.X_train_unlabeled.loc[certain_indices]
                    recommended_labels = [
                        frequencies.most_common(1)[0][0] for _ in certain_indices
                    ]
                    recommended_labels = pd.DataFrame(
                        recommended_labels, index=certain_X.index
                    )
                    #  log_it("Cluster ", cluster_id, certain_indices)
                    cluster_found = True
                    break

        # delete this cluster from the list of possible cluster for the next round
        if cluster_found:
            self.data_storage.X_train_labeled_cluster_indices.pop(cluster_id)
        return certain_X, recommended_labels, certain_indices

    def get_newly_labeled_data(self):
        X_train_unlabeled_cluster_indices = self.cluster_strategy.get_cluster_indices(
            clf=self.clf, nr_queries_per_iteration=self.nr_queries_per_iteration
        )

        # ask strategy for new datapoint
        query_indices = self.calculate_next_query_indices(
            X_train_unlabeled_cluster_indices
        )

        X_query = self.data_storage.X_train_unlabeled.loc[query_indices]

        # ask oracle for new query
        Y_query = self.oracle.get_labels(query_indices, self.data_storage)
        return X_query, Y_query, query_indices

    def uncertainty_recommendation(self, CERTAINTY_THRESHOLD, CERTAINTY_RATIO):
        # calculate certainties for all of X_train_unlabeled
        certainties = self.clf.predict_proba(
            self.data_storage.X_train_unlabeled.to_numpy()
        )

        amount_of_certain_labels = np.count_nonzero(
            np.where(np.max(certainties, 1) > CERTAINTY_THRESHOLD)
        )

        if (
            amount_of_certain_labels
            > len(self.data_storage.X_train_unlabeled) * CERTAINTY_RATIO
        ):

            # for safety reasons I refrain from explaining the following
            certain_indices = [
                j
                for i, j in enumerate(
                    self.data_storage.X_train_unlabeled.index.tolist()
                )
                if np.max(certainties, 1)[i] > CERTAINTY_THRESHOLD
            ]

            certain_X = self.data_storage.X_train_unlabeled.loc[certain_indices]

            recommended_labels = self.clf.predict(certain_X.to_numpy())
            # add indices to recommended_labels, could be maybe useful later on?
            recommended_labels = pd.DataFrame(recommended_labels, index=certain_X.index)

            return certain_X, recommended_labels, certain_indices
        else:
            return None, None, None

    def snuba_lite_recommendation(self, MINIMUM_HEURISTIC_ACCURACY):

        X_weak = Y_weak = weak_indices = None
        # @todo prevent snuba_lite from relearning based on itself (so only "strong" labels are being used for weak labeling)
        # for each label and each feature (or feature combination) generate small shallow decision tree -> is it a good idea to limit the amount of used features?!
        highest_accuracy = 0
        best_heuristic = None
        best_combination = None
        best_class = None

        combinations = []
        for combination in itertools.combinations(
            list(range(self.data_storage.X_train_labeled.shape[1])), 1
        ):
            combinations.append(combination)

        # generated heuristics should only being applied to small subset (which one?)
        # balance jaccard and f1_measure (coverage + accuracy)
        for clf_class in self.data_storage.label_encoder.classes_:
            for combination in combinations:
                # create training and test data set out of current available training/test data
                X_temp = self.data_storage.X_train_labeled.loc[:, combination]

                # do one vs rest
                Y_temp = self.data_storage.Y_train_labeled.copy()
                Y_temp = Y_temp.replace(
                    self.data_storage.label_encoder.transform(
                        [
                            c
                            for c in self.data_storage.label_encoder.classes_
                            if c != clf_class
                        ]
                    ),
                    -1,
                )

                X_temp_train, X_temp_test, Y_temp_train, Y_temp_test = train_test_split(
                    X_temp, Y_temp, train_size=0.6
                )

                heuristic = DecisionTreeClassifier(max_depth=2)
                heuristic.fit(X_temp_train, Y_temp_train)

                accuracy = accuracy_score(Y_temp_test, heuristic.predict(X_temp_test))
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_heuristic = heuristic
                    best_combination = combination
                    best_class = clf_class

        # if accuracy of decision tree is high enough -> take recommendation
        if highest_accuracy > MINIMUM_HEURISTIC_ACCURACY:
            probabilities = best_heuristic.predict_proba(
                self.data_storage.X_train_unlabeled.loc[:, best_combination].to_numpy()
            )

            # filter out labels where one-vs-rest heuristic is sure that sample is of label L
            weak_indices = [
                index
                for index, proba in zip(
                    self.data_storage.X_train_unlabeled.index, probabilities
                )
                if np.max(proba) > MINIMUM_HEURISTIC_ACCURACY
            ]

            if len(weak_indices) > 0:
                log_it("Snuba mit Klasse " + best_class)
                #  log_it(weak_indices)
                X_weak = self.data_storage.X_train_unlabeled.loc[weak_indices]
                best_class_encoded = self.data_storage.label_encoder.transform(
                    [best_class]
                )[0]
                Y_weak = [best_class_encoded for _ in weak_indices]
            else:
                weak_indices = None

        return X_weak, Y_weak, weak_indices

    def learn(
        self,
        MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS,
        WITH_CLUSTER_RECOMMENDATION,
        WITH_UNCERTAINTY_RECOMMENDATION,
        WITH_SNUBA_LITE,
        ALLOW_RECOMMENDATIONS_AFTER_STOP,
        USER_QUERY_BUDGET_LIMIT,
        **kwargs,
    ):
        log_it(self.data_storage.label_encoder.classes_)
        log_it("Used Hyperparams:")
        log_it(vars(self))
        log_it(locals())

        log_it(get_single_al_run_stats_table_header())

        self.start_set_size = len(self.data_storage.ground_truth_indices)
        early_stop_reached = False

        for i in range(0, self.NR_LEARNING_ITERATIONS):
            # try to actively get at least this amount of data, but if there is only less data available that's just fine
            if (
                self.data_storage.X_train_unlabeled.shape[0]
                < self.nr_queries_per_iteration
            ):
                self.nr_queries_per_iteration = self.data_storage.X_train_unlabeled.shape[
                    0
                ]
            if self.nr_queries_per_iteration == 0:
                break

            # first iteration - add everything from ground truth
            if i == 0:
                query_indices = self.data_storage.ground_truth_indices
                X_query = self.data_storage.X_train_unlabeled.loc[query_indices]
                Y_query = self.data_storage.Y_train_unlabeled.loc[query_indices]

                recommendation_value = "G"
                Y_query_strong = None
            else:
                X_query = None

                if (
                    self.metrics_per_al_cycle["test_acc"][-1]
                    > MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS
                ):
                    if X_query is None and WITH_CLUSTER_RECOMMENDATION:
                        X_query, Y_query, query_indices = self.cluster_recommendation(
                            kwargs["CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE"],
                            kwargs["CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED"],
                        )
                        recommendation_value = "C"

                    if X_query is None and WITH_UNCERTAINTY_RECOMMENDATION:
                        (
                            X_query,
                            Y_query,
                            query_indices,
                        ) = self.uncertainty_recommendation(
                            kwargs["UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"],
                            kwargs["UNCERTAINTY_RECOMMENDATION_RATIO"],
                        )
                        recommendation_value = "U"

                    if X_query is None and WITH_SNUBA_LITE:
                        (
                            X_query,
                            Y_query,
                            query_indices,
                        ) = self.snuba_lite_recommendation(
                            kwargs["SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY"]
                        )
                        recommendation_value = "S"

                if early_stop_reached and X_query is None:
                    break

                if X_query is None:
                    # ask oracle for some new labels
                    X_query, Y_query, query_indices = self.get_newly_labeled_data()
                    recommendation_value = "A"
                    self.amount_of_user_asked_queries += len(Y_query)

            Y_query = Y_query.assign(source=recommendation_value)

            self.metrics_per_al_cycle["recommendation"].append(recommendation_value)
            self.metrics_per_al_cycle["query_length"].append(len(Y_query))
            self.metrics_per_al_cycle["labels_indices"].append(str(query_indices))

            self.data_storage.move_labeled_queries(X_query, Y_query, query_indices)

            self.calculate_pre_metrics(X_query, Y_query)

            # retrain CLASSIFIER
            self.fit_clf()

            self.calculate_post_metrics(X_query, Y_query)

            log_it(
                get_single_al_run_stats_row(
                    i,
                    self.data_storage.X_train_labeled.shape[0],
                    self.data_storage.X_train_unlabeled.shape[0],
                    self.metrics_per_al_cycle,
                )
            )

            if self.amount_of_user_asked_queries > USER_QUERY_BUDGET_LIMIT:
                early_stop_reached = True
                log_it("Budget exhausted")
                if not ALLOW_RECOMMENDATIONS_AFTER_STOP:
                    break

        return (
            self.clf,
            self.metrics_per_al_cycle,
            self.data_storage.Y_train_labeled,
        )
