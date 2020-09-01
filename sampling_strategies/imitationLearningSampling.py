import statistics
import copy
import random
from itertools import chain

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from ..activeLearner import ActiveLearner


def sample_unlabeled_X(
    train_unlabeled_X, train_labeled_X, sample_size, CONVEX_HULL_SAMPLING
):
    if CONVEX_HULL_SAMPLING:
        max_sum = 0
        for i in range(0, 100):
            random_sample = train_unlabeled_X.sample(n=sample_size)

            # calculate distance to each other
            total_distance = np.sum(pairwise_distances(random_sample, random_sample))
            #  total_distance += np.sum(
            #      pairwise_distances(random_sample, train_unlabeled_X)
            #  )
            total_distance += np.sum(pairwise_distances(random_sample, train_labeled_X))
            if total_distance > max_sum:
                max_sum = total_distance
                X_query = random_sample
        possible_samples_indices = X_query.index

    else:
        X_query = train_unlabeled_X.sample(n=sample_size)
    return X_query


def _future_peak(
    unlabeled_sample_indice,
    weak_supervision_label_sources,
    data_storage,
    clf,
    MAX_AMOUNT_OF_WS_PEAKS,
):
    copy_of_data_storage = copy.deepcopy(data_storage)
    copy_of_classifier = copy.deepcopy(clf)

    copy_of_data_storage.label_samples(
        pd.Index([unlabeled_sample_indice]),
        [copy_of_data_storage.train_unlabeled_Y.loc[unlabeled_sample_indice]["label"]],
        "P",
    )
    copy_of_classifier.fit(
        copy_of_data_storage.train_labeled_X,
        copy_of_data_storage.train_labeled_Y["label"].to_list(),
    )
    for labelSource in weak_supervision_label_sources:
        labelSource.data_storage = copy_of_data_storage

    # what would happen if we apply WS after this one?
    for i in range(0, MAX_AMOUNT_OF_WS_PEAKS):
        for labelSource in weak_supervision_label_sources:
            (
                Y_query,
                query_indices,
                source,
            ) = labelSource.get_labeled_samples()

            if Y_query is not None:
                break
        if Y_query is None:
            ws_still_applicable = False
            continue

        copy_of_data_storage.label_samples(query_indices, Y_query, source)

        copy_of_classifier.fit(
            copy_of_data_storage.train_labeled_X,
            copy_of_data_storage.train_labeled_Y["label"].to_list(),
        )

    Y_pred_train_unlabeled = copy_of_classifier.predict(
        copy_of_data_storage.train_unlabeled_X
    )
    Y_pred_test = copy_of_classifier.predict(copy_of_data_storage.test_X)
    Y_pred = np.concatenate((Y_pred_train_unlabeled, Y_pred_test))
    Y_true = (
        copy_of_data_storage.train_unlabeled_Y["label"].to_list()
        + copy_of_data_storage.test_Y["label"].to_list()
    )

    Y_pred = Y_pred_test
    Y_true = copy_of_data_storage.test_Y["label"].to_list()

    accuracy_with_that_label = accuracy_score(Y_pred, Y_true)

    #  print(
    #      "Testing out : {}, train acc: {}".format(
    #          unlabeled_sample_indice, accuracy_with_that_label
    #      )
    #  )
    return accuracy_with_that_label


def calculate_state(
    X_query,
    data_storage,
    clf,
    OLD=False,
    NO_DIFF_FEATURES=False,
    LRU_AREAS_LIMIT=0,
    lru_samples=[],
):
    possible_samples_probas = clf.predict_proba(X_query)

    sorted_probas = -np.sort(-possible_samples_probas, axis=1)
    argmax_probas = sorted_probas[:, 0]
    argsecond_probas = sorted_probas[:, 1]

    if OLD:
        if NO_DIFF_FEATURES:
            arg_diff_probas = argmax_probas - argsecond_probas
            argsecond_probas = arg_diff_probas
        return np.array([*argmax_probas, *argsecond_probas])

    arg_diff_probas = argmax_probas - argsecond_probas

    if not NO_DIFF_FEATURES:
        arg_diff_probas = argsecond_probas

    if LRU_AREAS_LIMIT == 0:
        # calculate average distance to labeled and average distance to unlabeled samples
        average_distance_labeled = (
            np.sum(
                pairwise_distances(data_storage.train_labeled_X, X_query),
                axis=0,
            )
            / len(data_storage.train_labeled_X)
        )
        average_distance_unlabeled = (
            np.sum(
                pairwise_distances(data_storage.train_unlabeled_X, X_query),
                axis=0,
            )
            / len(data_storage.train_unlabeled_X)
        )

        X_state = np.array(
            [
                *argmax_probas,
                *arg_diff_probas,
                *average_distance_labeled,
                *average_distance_unlabeled,
            ]
        )
    else:
        if len(lru_samples) == 0:
            lru_distances = [0 for _ in range(0, len(X_query))]
        else:
            lru_distances = (
                np.sum(
                    np.multiply(
                        [i for i in range(1, len(lru_samples) + 1)],
                        pairwise_distances(X_query, lru_samples),
                    ),
                    axis=1,
                )
                / len(lru_samples)
            )
        X_state = np.array([*argmax_probas, *arg_diff_probas, *lru_distances])
    return X_state


class ImitationLearner(ActiveLearner):
    def set_amount_of_peaked_objects(self, amount_of_peaked_objects):
        self.amount_of_peaked_objects = amount_of_peaked_objects

    def init_sampling_classifier(
        self,
        DATA_PATH,
        REPRESENTATIVE_FEATURES,
        CONVEX_HULL_SAMPLING,
        VARIANCE_BOUND,
        NO_DIFF_FEATURES,
        LRU_AREAS_LIMIT,
    ):
        self.states = pd.DataFrame(
            data=None,
        )
        self.optimal_policies = pd.DataFrame(
            data=None,
            columns=[
                str(i) + "_true_peaked_normalised_acc"
                for i in range(0, self.amount_of_peaked_objects)
            ],
        )

        self.REPRESENTATIVE_FEATURES = REPRESENTATIVE_FEATURES
        self.CONVEX_HULL_SAMPLING = CONVEX_HULL_SAMPLING
        self.VARIANCE_BOUND = VARIANCE_BOUND
        self.NO_DIFF_FEATURES = NO_DIFF_FEATURES
        self.LRU_AREAS_LIMIT = LRU_AREAS_LIMIT
        self.lru_samples = pd.DataFrame(
            data=None, columns=self.data_storage.train_unlabeled_X.columns, index=None
        )

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.train_labeled_X = self.train_labeled_X.append(X_query)
        self.train_unlabeled_X = self.train_unlabeled_X.drop(query_indices)

        try:
            self.Y_train_strong_labels = self.Y_train_strong_labels.append(
                self.Y_train_unlabeled.loc[query_indices]
            )
        except KeyError:
            # in a non experiment setting an error will be thrown because self.Y_train_unlabeled of course doesn't contains the labels
            for query_index in query_indices:
                self.Y_train_strong_labels.loc[query_index] = [-1]

        self.Y_train_labeled = self.Y_train_labeled.append(Y_query)
        self.Y_train_unlabeled = self.Y_train_unlabeled.drop(
            query_indices, errors="ignore"
        )

        # remove indices from all clusters in unlabeled and add to labeled
        for cluster_id in self.train_unlabeled_X_cluster_indices.keys():
            list_to_be_removed_and_appended = []
            for indice in query_indices:
                if indice in self.train_unlabeled_X_cluster_indices[cluster_id]:
                    list_to_be_removed_and_appended.append(indice)

            # don't change a list you're iterating over!
            for indice in list_to_be_removed_and_appended:
                self.train_unlabeled_X_cluster_indices[cluster_id].remove(indice)
                self.train_labeled_X_cluster_indices[cluster_id].append(indice)

        # remove possible empty clusters
        self.train_unlabeled_X_cluster_indices = {
            k: v
            for k, v in self.train_unlabeled_X_cluster_indices.items()
            if len(v) != 0
        }

    def save_nn_training_data(self, DATA_PATH):
        self.states.to_csv(
            DATA_PATH + "/states.csv", index=False, header=False, mode="a"
        )
        self.optimal_policies.to_csv(
            DATA_PATH + "/opt_pol.csv", index=False, header=False, mode="a"
        )

    """
    We take a "peak" into the future and annotate exactly those samples where we KNOW that they will benefit us the most
    """

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        if len(self.metrics_per_al_cycle["test_acc"]) > 1:
            if (
                self.metrics_per_al_cycle["test_acc"][-2]
                >= self.metrics_per_al_cycle["test_acc"][-1]
            ):
                self.states = self.states.head(-1)
                self.optimal_policies = self.optimal_policies.head(-1)
                self.data_storage.deleted = True
            else:
                self.data_storage.deleted = False
        # merge indices from all clusters together and take the n most uncertain ones from them
        #  train_unlabeled_X_indices = list(
        #      chain(*list(train_unlabeled_X_cluster_indices.values()))
        #  )

        future_peak_acc = []

        #  random.shuffle(train_unlabeled_X_indices)
        good_sample_found = False
        hard_kill_count = 0
        best_largest_stdev = 0
        best_future_peak_acc = None
        best_possible_samples_X = None
        best_possible_sample_indices = None

        while not good_sample_found and hard_kill_count < self.VARIANCE_BOUND:
            possible_samples_X = sample_unlabeled_X(
                self.data_storage.train_unlabeled_X,
                self.data_storage.train_labeled_X,
                self.amount_of_peaked_objects,
                self.CONVEX_HULL_SAMPLING,
            )
            possible_samples_indices = possible_samples_X.index

            # parallelisieren
            with parallel_backend("loky", n_jobs=self.N_JOBS):
                future_peak_acc = Parallel()(
                    delayed(_future_peak)(
                        unlabeled_sample_indice,
                        self.weak_supervision_label_sources,
                        self.data_storage,
                        self.clf,
                        self.MAX_AMOUNT_OF_WS_PEAKS,
                    )
                    for unlabeled_sample_indice in possible_samples_indices
                )
            if max(future_peak_acc) > best_largest_stdev:
                best_future_peak_acc = future_peak_acc
                best_possible_sample_indices = possible_samples_indices
                best_possible_samples_X = possible_samples_X

            if statistics.stdev(future_peak_acc) > 0.03:
                good_sample_found = True
            else:
                hard_kill_count += 1

        if hard_kill_count == self.VARIANCE_BOUND:
            future_peak_acc = best_future_peak_acc
            possible_sample_indices = best_possible_sample_indices
            possible_samples_X = best_possible_samples_X

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.possible_samples_indices = possible_samples_indices

        for labelSource in self.weak_supervision_label_sources:
            labelSource.data_storage = self.data_storage

        X_state = calculate_state(
            possible_samples_X,
            self.data_storage,
            self.clf,
            OLD=self.REPRESENTATIVE_FEATURES,
            LRU_AREAS_LIMIT=self.LRU_AREAS_LIMIT,
            NO_DIFF_FEATURES=self.NO_DIFF_FEATURES,
            lru_samples=self.lru_samples,
        )

        # take first and second most examples from possible_samples_probas and append them then to states
        self.states = self.states.append(
            pd.Series(X_state),
            ignore_index=True
            #  pd.Series(dict(zip(self.states.columns, X_state))), ignore_index=True,
        )

        # the output of the net is one neuron per possible unlabeled sample, and the output should be:
        # a) 0 for label this, do not label that (how to determine how many, how to forbid how many labeled samples are allowed?)
        # b) predict accuracy gain by labeling sample x -> I can compare it directly to the values I could predict -> normalize values as 0 -> smallest possible gain, 1 -> highest possible gain
        #  future_peak_accs = [[b] for b in future_peak_acc]
        future_peak_accs = future_peak_acc
        # min max scaling of output
        #  scaler = MinMaxScaler()
        #  future_peak_accs = [a[0] for a in scaler.fit_transform(future_peak_accs)]

        # save the indices of the n_best possible states, order doesn't matter
        self.optimal_policies = self.optimal_policies.append(
            pd.Series(dict(zip(self.optimal_policies.columns, future_peak_accs))),
            ignore_index=True,
        )

        sorting = self.optimal_policies.iloc[-1, :].to_numpy()

        # use the optimal values
        zero_to_one_values_and_index = list(zip(sorting, possible_samples_indices))
        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        if self.LRU_AREAS_LIMIT > 0:
            # fifo queue
            self.lru_samples = pd.concat(
                [
                    self.lru_samples,
                    self.data_storage.train_unlabeled_X.loc[
                        [
                            v
                            for k, v in ordered_list_of_possible_sample_indices[
                                : self.nr_queries_per_iteration
                            ]
                        ]
                    ],
                ],
                #  ignore_index=True,
            )

            # clean up
            if len(self.lru_samples) > self.LRU_AREAS_LIMIT:
                self.lru_samples = self.lru_samples.tail(self.LRU_AREAS_LIMIT)

        return [
            v
            for k, v in ordered_list_of_possible_sample_indices[
                : self.nr_queries_per_iteration
            ]
        ]
