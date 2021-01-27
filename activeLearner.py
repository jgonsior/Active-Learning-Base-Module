import abc

from .experiment_setup_lib import (
    conf_matrix_and_acc_and_f1,
    get_single_al_run_stats_row,
    get_single_al_run_stats_table_header,
    log_it,
)


class ActiveLearner:
    def __init__(
        self,
        data_storage,
        cluster_strategy,
        oracle,
        clf,
        weak_supervision_label_sources=[],
        **kwargs,
    ):

        self.__dict__.update(**kwargs)

        self.data_storage = data_storage
        self.clf = clf

        self.metrics_per_al_cycle = {
            "test_acc": [],
            "test_f1": [],
            "test_conf_matrix": [],
            "train_acc": [],
            "train_f1": [],
            "train_conf_matrix": [],
            "query_length": [],
            "source": [],
        }
        self.cluster_strategy = cluster_strategy
        self.amount_of_user_asked_queries = 0
        self.oracle = oracle
        self.weak_supervision_label_sources = weak_supervision_label_sources

        # fake iteration zero
        X_query = self.data_storage.X[self.data_storage.labeled_mask]
        Y_query = self.data_storage.Y[self.data_storage.labeled_mask]

        self.metrics_per_al_cycle["source"].append("G")
        self.metrics_per_al_cycle["query_length"].append(len(Y_query))

        self.calculate_pre_metrics(X_query, Y_query)

        # retrain CLASSIFIER
        self.fit_clf()

        self.calculate_post_metrics(X_query, Y_query)

        log_it(get_single_al_run_stats_table_header())
        log_it(
            get_single_al_run_stats_row(
                0,
                len(self.data_storage.labeled_mask),
                len(self.data_storage.unlabeled_mask),
                self.metrics_per_al_cycle,
            )
        )

    @abc.abstractmethod
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        pass

    def fit_clf(self):
        self.clf.fit(
            self.data_storage.X[self.data_storage.labeled_mask],
            self.data_storage.Y[self.data_storage.labeled_mask],
            #  sample_weight=compute_sample_weight(
            #      "balanced",
            #      self.data_storage.Y[self.data_storage.labeled_mask],
            #  ),
        )

    def calculate_pre_metrics(self, X_query, Y_query):
        pass

    def calculate_post_metrics(self, X_query, Y_query):
        if len(self.data_storage.test_mask) > 0:
            # experiment
            conf_matrix, acc, f1 = conf_matrix_and_acc_and_f1(
                self.clf,
                self.data_storage.X[self.data_storage.test_mask],
                self.data_storage.Y[self.data_storage.test_mask],
                self.data_storage.label_encoder,
            )
        else:
            conf_matrix, acc, f1 = None, 0, 0
        self.metrics_per_al_cycle["test_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["test_acc"].append(acc)
        self.metrics_per_al_cycle["test_f1"].append(f1)

        if len(self.data_storage.test_mask) > 0:
            # experiment
            conf_matrix, acc, f1 = conf_matrix_and_acc_and_f1(
                self.clf,
                self.data_storage.X[self.data_storage.labeled_mask],
                self.data_storage.Y[self.data_storage.labeled_mask],
                self.data_storage.label_encoder,
            )
        else:
            conf_matrix, acc, f1 = None, 0, 0

        self.metrics_per_al_cycle["train_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["train_acc"].append(acc)
        self.metrics_per_al_cycle["train_f1"].append(f1)

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.train_unlabeled_Y_predicted = self.clf.predict(
                self.data_storage.X[self.data_storage.unlabeled_mask]
            )
            self.data_storage.train_labeled_Y_predicted = self.clf.predict(
                self.data_storage.X[self.data_storage.labeled_mask]
            )

    def get_newly_labeled_data(self):
        X_train_unlabeled_cluster_indices = self.cluster_strategy.get_cluster_indices(
            clf=self.clf, nr_queries_per_iteration=self.NR_QUERIES_PER_ITERATION
        )

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.possible_samples_indices = []
            self.data_storage.test_accuracy = self.metrics_per_al_cycle["test_acc"][-1]
            self.data_storage.clf = self.clf

        # ask strategy for new datapoint
        query_indices = self.calculate_next_query_indices(
            X_train_unlabeled_cluster_indices
        )

        # ask oracle for new query
        Y_query = self.oracle.get_labeled_samples(
            query_indices, self.data_storage, self.metrics_per_al_cycle
        )
        return Y_query, query_indices, "A"

    def learn(
        self,
    ):
        log_it(self.data_storage.label_encoder.classes_)
        log_it("Used Hyperparams:")
        log_it(vars(self))
        log_it(locals())

        early_stop_reached = False
        for i in range(0, self.NR_LEARNING_ITERATIONS):
            # try to actively get at least this amount of data, but if there is only less data available that's just fine as well
            if len(self.data_storage.unlabeled_mask) < self.NR_QUERIES_PER_ITERATION:
                self.NR_QUERIES_PER_ITERATION = len(self.data_storage.unlabeled_mask)
            if self.NR_QUERIES_PER_ITERATION == 0:
                break

            # first iteration - add everything from ground truth
            Y_query = None

            if (
                self.metrics_per_al_cycle["test_acc"][-1]
                > self.MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS
            ):
                # iterate over existing WS sources
                for labelSource in self.weak_supervision_label_sources:
                    (
                        Y_query,
                        query_indices,
                        source,
                    ) = labelSource.get_labeled_samples()

                    if Y_query is not None:
                        break

            if early_stop_reached and Y_query is None:
                break

            if Y_query is None:
                # ask oracle for some new labels
                Y_query, query_indices, source = self.get_newly_labeled_data()
                self.amount_of_user_asked_queries += len(Y_query)

            self.metrics_per_al_cycle["source"].append(source)
            self.metrics_per_al_cycle["query_length"].append(len(Y_query))

            self.data_storage.label_samples(query_indices, Y_query, source)

            X_query = self.data_storage.X[query_indices]

            self.calculate_pre_metrics(X_query, Y_query)

            # retrain CLASSIFIER
            self.fit_clf()

            self.calculate_post_metrics(X_query, Y_query)

            log_it(
                get_single_al_run_stats_row(
                    i,
                    len(self.data_storage.labeled_mask),
                    len(self.data_storage.unlabeled_mask),
                    self.metrics_per_al_cycle,
                )
            )

            if self.amount_of_user_asked_queries > self.USER_QUERY_BUDGET_LIMIT:
                early_stop_reached = True
                log_it("Budget exhausted")
                if not self.ALLOW_RECOMMENDATIONS_AFTER_STOP:
                    break

            if self.STOP_AFTER_MAXIMUM_ACCURACY_REACHED:
                if (
                    self.metrics_per_al_cycle["test_acc"][-1]
                    >= self.THEORETICALLY_BEST_ACHIEVABLE_ACCURACY
                ):
                    early_stop_reached = True
                    log_it(
                        "THEORETICALLY_BEST_ACHIEVABLE_ACCURACY: "
                        + str(self.THEORETICALLY_BEST_ACHIEVABLE_ACCURACY)
                    )
                    break

        return (self.clf, self.metrics_per_al_cycle)
