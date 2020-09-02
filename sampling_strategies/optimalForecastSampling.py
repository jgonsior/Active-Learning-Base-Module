import copy
import random
from itertools import chain

import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.metrics import accuracy_score

from ..activeLearner import ActiveLearner


class OptimalForecastSampler(ActiveLearner):
    def set_amount_of_peaked_objects(self, amount_of_peaked_objects):
        self.amount_of_peaked_objects = amount_of_peaked_objects

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

    def _future_peak(self, unlabeled_sample_indice):
        copy_of_data_storage = copy.deepcopy(self.data_storage)
        copy_of_classifier = copy.deepcopy(self.clf)

        copy_of_data_storage.label_samples(
            pd.Index([unlabeled_sample_indice]),
            [
                copy_of_data_storage.train_unlabeled_Y.loc[unlabeled_sample_indice][
                    "label"
                ]
            ],
            "P",
        )
        copy_of_classifier.fit(
            copy_of_data_storage.train_labeled_X,
            copy_of_data_storage.train_labeled_Y["label"].to_list(),
        )
        for labelSource in self.weak_supervision_label_sources:
            labelSource.data_storage = copy_of_data_storage

        # what would happen if we apply WS after this one?
        for i in range(0, self.MAX_AMOUNT_OF_WS_PEAKS):
            for labelSource in self.weak_supervision_label_sources:
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

        Y_pred = copy_of_classifier.predict(copy_of_data_storage.train_unlabeled_X)

        accuracy_with_that_label = accuracy_score(
            Y_pred, copy_of_data_storage.train_unlabeled_Y["label"].to_list()
        )

        #  print(
        #      "Testing out : {}, acc: {}".format(
        #          unlabeled_sample_indice, accuracy_with_that_label
        #      )
        #  )
        return unlabeled_sample_indice, accuracy_with_that_label

    """
    We take a "peak" into the future and annotate exactly those samples where we KNOW that they will benefit us the most
    """

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        train_unlabeled_X_indices = list(
            chain(*list(train_unlabeled_X_cluster_indices.values()))
        )

        scores = []

        random.shuffle(train_unlabeled_X_indices)

        # parallelisieren
        with parallel_backend("loky", n_jobs=self.N_JOBS):
            scores = Parallel()(
                delayed(self._future_peak)(unlabeled_sample_indice)
                for unlabeled_sample_indice in train_unlabeled_X_indices[
                    : self.amount_of_peaked_objects
                ]
            )
        for labelSource in self.weak_supervision_label_sources:
            labelSource.data_storage = self.data_storage

        scores = sorted(scores, key=lambda tup: tup[1], reverse=True)
        return [k for k, v in scores[: self.nr_queries_per_iteration]]
