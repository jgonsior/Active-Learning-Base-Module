import abc

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
        data_storage,
        cluster_strategy,
        N_JOBS,
        NR_LEARNING_ITERATIONS,
        NR_QUERIES_PER_ITERATION,
        oracle,
        clf,
        weak_supervision_label_sources=[],
    ):
        self.data_storage = data_storage
        self.NR_LEARNING_ITERATIONS = NR_LEARNING_ITERATIONS
        self.nr_queries_per_iteration = NR_QUERIES_PER_ITERATION
        self.clf = clf

        self.metrics_per_al_cycle = {
            "test_acc": [],
            "test_conf_matrix": [],
            "train_acc": [],
            "train_conf_matrix": [],
            "query_length": [],
            "source": [],
            "labels_indices": [],
        }
        self.cluster_strategy = cluster_strategy
        self.amount_of_user_asked_queries = 0
        self.oracle = oracle
        self.weak_supervision_label_sources = weak_supervision_label_sources
        self.N_JOBS = N_JOBS
        self.RANDOM_SEED = RANDOM_SEED

        # fake iteration zero
        X_query = self.data_storage.train_labeled_X
        Y_query = self.data_storage.train_labeled_Y

        self.metrics_per_al_cycle["source"].append("G")
        self.metrics_per_al_cycle["query_length"].append(len(Y_query))
        self.metrics_per_al_cycle["labels_indices"].append(str(X_query.index))

        self.calculate_pre_metrics(X_query, Y_query)

        # retrain CLASSIFIER
        self.fit_clf()

        self.calculate_post_metrics(X_query, Y_query)

        log_it(get_single_al_run_stats_table_header())
        log_it(
            get_single_al_run_stats_row(
                0,
                len(self.data_storage.train_labeled_X),
                len(self.data_storage.train_unlabeled_X),
                self.metrics_per_al_cycle,
            )
        )

    @abc.abstractmethod
    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        pass

    def fit_clf(self):
        self.clf.fit(
            self.data_storage.train_labeled_X,
            self.data_storage.train_labeled_Y["label"].to_list(),
            sample_weight=compute_sample_weight(
                "balanced", self.data_storage.train_labeled_Y["label"].to_list(),
            ),
        )

    def calculate_pre_metrics(self, X_query, Y_query):
        pass

    def calculate_post_metrics(self, X_query, Y_query):
        if len(self.data_storage.test_X) > 0:
            # experiment
            conf_matrix, acc = conf_matrix_and_acc(
                self.clf,
                self.data_storage.test_X,
                self.data_storage.test_Y["label"].to_list(),
                self.data_storage.label_encoder,
            )
        else:
            conf_matrix, acc = None, 0

        self.metrics_per_al_cycle["test_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["test_acc"].append(acc)

        if not self.data_storage.train_unlabeled_Y.label.isnull().values.any():
            # experiment
            conf_matrix, acc = conf_matrix_and_acc(
                self.clf,
                self.data_storage.train_labeled_X,
                self.data_storage.train_labeled_Y["label"].to_list(),
                self.data_storage.label_encoder,
            )
        else:
            conf_matrix, acc = None, 0

        self.metrics_per_al_cycle["train_conf_matrix"].append(conf_matrix)
        self.metrics_per_al_cycle["train_acc"].append(acc)

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.train_unlabeled_Y_predicted = self.clf.predict(
                self.data_storage.train_unlabeled_X
            )
            self.data_storage.train_labeled_Y_predicted = self.clf.predict(
                self.data_storage.train_labeled_X
            )

    def get_newly_labeled_data(self):
        X_train_unlabeled_cluster_indices = self.cluster_strategy.get_cluster_indices(
            clf=self.clf, nr_queries_per_iteration=self.nr_queries_per_iteration
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
        Y_query = self.oracle.get_labeled_samples(query_indices, self.data_storage)
        return Y_query, query_indices, "A"

    def learn(
        self,
        MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS,
        ALLOW_RECOMMENDATIONS_AFTER_STOP,
        USER_QUERY_BUDGET_LIMIT,
        **kwargs,
    ):
        log_it(self.data_storage.label_encoder.classes_)
        log_it("Used Hyperparams:")
        log_it(vars(self))
        log_it(locals())

        early_stop_reached = False

        for i in range(0, self.NR_LEARNING_ITERATIONS):
            # try to actively get at least this amount of data, but if there is only less data available that's just fine as well
            if len(self.data_storage.train_unlabeled_X) < self.nr_queries_per_iteration:
                self.nr_queries_per_iteration = len(self.data_storage.train_unlabeled_X)
            if self.nr_queries_per_iteration == 0:
                break

            # first iteration - add everything from ground truth
            Y_query = None

            if (
                self.metrics_per_al_cycle["test_acc"][-1]
                > MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS
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
            self.metrics_per_al_cycle["labels_indices"].append(str(query_indices))

            self.data_storage.label_samples(query_indices, Y_query, source)

            X_query = self.data_storage.train_labeled_X.loc[query_indices]

            self.calculate_pre_metrics(X_query, Y_query)

            # retrain CLASSIFIER
            self.fit_clf()

            self.calculate_post_metrics(X_query, Y_query)

            log_it(
                get_single_al_run_stats_row(
                    i,
                    len(self.data_storage.train_labeled_X),
                    len(self.data_storage.train_unlabeled_X),
                    self.metrics_per_al_cycle,
                )
            )

            if self.amount_of_user_asked_queries > USER_QUERY_BUDGET_LIMIT:
                early_stop_reached = True
                log_it("Budget exhausted")
                if not ALLOW_RECOMMENDATIONS_AFTER_STOP:
                    break
            if "STOP_AFTER_MAXIMUM_ACCURACY_REACHED" in kwargs:
                if kwargs["STOP_AFTER_MAXIMUM_ACCURACY_REACHED"]:
                    if (
                        self.metrics_per_al_cycle["test_acc"][-1]
                        >= kwargs["THEORETICALLY_BEST_ACHIEVABLE_ACCURACY"]
                    ):
                        early_stop_reached = True
                        print(
                            "THEORETICALLY_BEST_ACHIEVABLE_ACCURACY: "
                            + str(kwargs["THEORETICALLY_BEST_ACHIEVABLE_ACCURACY"])
                        )
                        break

        return (self.clf, self.metrics_per_al_cycle)
