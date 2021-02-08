import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .learnedBaseBatchSampling import LearnedBaseBatchSampling


class ImitationBatchLearner(LearnedBaseBatchSampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = pd.DataFrame(
            data=None,
        )
        self.optimal_policies = pd.DataFrame(
            data=None,
            columns=[
                str(i) + "_true_peaked_normalised_acc"
                for i in range(0, self.AMOUNT_OF_PEAKED_OBJECTS)
            ],
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

    def calculate_next_query_indices_pre_hook(self):
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

    def get_X_query_index(self):
        future_peak_acc = []

        index_batches = self.pre_sample_potential_X_queries(
            self.AMOUNT_OF_PEAKED_OBJECTS,
            self.PRE_SAMPLING_METHOD,
            self.PRE_SAMPLING_ARG,
        )

        future_peak_acc = []
        # single thread
        for index_batch in index_batches:
            future_peak_acc.append(
                self._future_peak(
                    index_batch,
                    self.weak_supervision_label_sources,
                    self.data_storage,
                    self.clf,
                    self.MAX_AMOUNT_OF_WS_PEAKS,
                )
            )

        for labelSource in self.weak_supervision_label_sources:
            labelSource.data_storage = self.data_storage

        self.optimal_policies = self.optimal_policies.append(
            pd.Series(dict(zip(self.optimal_policies.columns, future_peak_acc))),
            ignore_index=True,
        )
        return index_batches

    def calculate_next_query_indices_post_hook(self, X_state):
        self.states = self.states.append(
            pd.Series(X_state),
            ignore_index=True
            #  pd.Series(dict(zip(self.states.columns, X_state))), ignore_index=True,
        )

    def get_sorting(self, X_state):
        return self.optimal_policies.iloc[-1, :].to_numpy()

    def _future_peak(
        self,
        unlabeled_sample_indices,
        weak_supervision_label_sources,
        data_storage,
        clf,
        MAX_AMOUNT_OF_WS_PEAKS,
    ):
        copy_of_classifier = copy.deepcopy(clf)

        copy_of_labeled_mask = np.append(
            data_storage.labeled_mask, unlabeled_sample_indices, axis=0
        )

        copy_of_classifier.fit(
            data_storage.X[copy_of_labeled_mask], data_storage.Y[copy_of_labeled_mask]
        )

        Y_pred_test = copy_of_classifier.predict(data_storage.X)
        Y_true = data_storage.Y

        accuracy_with_that_label = accuracy_score(Y_pred_test, Y_true)

        return accuracy_with_that_label
