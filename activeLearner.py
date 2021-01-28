import abc
from sklearn.base import BaseEstimator
from oracles import BaseOracle
from callbacks import BaseCallback
from dataStorage import DataStorage
from stopping_criteria import BaseStoppingCriteria
from logger.logger import log_it


class ActiveLearner:
    def __init__(
        self,
        data_storage: DataStorage,
        oracles: List[BaseOracle],
        learner: BaseEstimator,
        callbacks: List[BaseCallback],
        stopping_criteria: BaseStoppingCriteria,
        **kwargs,
    ) -> None:

        self.__dict__.update(**kwargs)

        self.data_storage = data_storage
        self.learner = learner
        self.oracles = oracles
        self.callbacks = List[BaseCallback]
        self.stopping_criteria = stopping_criteria

        # fake iteration zero
        X_query = self.data_storage.X[self.data_storage.labeled_mask]
        Y_query = self.data_storage.Y[self.data_storage.labeled_mask]

        self.metrics_per_al_cycle["source"].append("G")
        self.metrics_per_al_cycle["query_length"].append(len(Y_query))

        self.calculate_pre_metrics(X_query, Y_query)

        # retrain CLASSIFIER
        self.fit_learner()

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

    def fit_learner(self):
        self.learner.fit(
            self.data_storage.X[self.data_storage.labeled_mask],
            self.data_storage.Y[self.data_storage.labeled_mask],
            #  sample_weight=compute_sample_weight(
            #      "balanced",
            #      self.data_storage.Y[self.data_storage.labeled_mask],
            #  ),
        )

    def get_newly_labeled_data(self):
        X_train_unlabeled_cluster_indices = self.cluster_strategy.get_cluster_indices(
            learner=self.learner, nr_queries_per_iteration=self.NR_QUERIES_PER_ITERATION
        )

        if self.data_storage.PLOT_EVOLUTION:
            self.data_storage.possible_samples_indices = []
            self.data_storage.test_accuracy = self.metrics_per_al_cycle["test_acc"][-1]
            self.data_storage.learner = self.learner

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
            self.fit_learner()

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

        return (self.learner, self.metrics_per_al_cycle)
