from typing import Dict, List, Union

from .callbacks.BaseCallback import BaseCallback
from .dataStorage import DataStorage, IndiceMask, LabelList
from .learner.standard import Learner
from .logger.logger import log_it
from .oracles.BaseOracle import BaseOracle
from .oracles.LabeledStartSetOracle import LabeledStartSetOracle
from .stopping_criterias.BaseStoppingCriteria import BaseStoppingCriteria


class ActiveLearner:
    query_sampling_strategy:         StandardQuerySamplingStrategy,
    data_storage: DataStorage
    oracle: BaseOracle
    callbacks: Dict[str, BaseCallback]
    stopping_criteria: BaseStoppingCriteria
    BATCH_SIZE: int
    current_query_indices: IndiceMask
    current_Y_queries: LabelList

    def __init__(
        self,
        query_sampling_strategy: StandardQuerySamplingStrategy,
        data_storage: DataStorage,
        oracle: BaseOracle,
        learner: Learner,
        callbacks: Dict[str, BaseCallback],
        stopping_criteria: BaseStoppingCriteria,
        BATCH_SIZE: int,
    ) -> None:

        self.query_sampling_strategy = query_sampling_strategy
        self.data_storage = data_storage
        self.learner = learner
        self.oracle = oracle
        self.callbacks = callbacks
        self.stopping_criteria = stopping_criteria
        self.BATCH_SIZE = BATCH_SIZE

        # fake iteration zero
        self.current_query_indices = self.data_storage.labeled_mask

        self.current_Y_queries = self.data_storage.Y[self.data_storage.labeled_mask]

        for callback in self.callbacks.values():
            callback.pre_learning_cycle_hook(self)

        # retrain CLASSIFIER
        self.fit_learner()

        # run all labeling functions to create weak labels
        self.data_storage.generate_weak_labels()

        for callback in self.callbacks.values():
            callback.post_learning_cycle_hook(self)

    def fit_learner(self) -> None:
        self.learner.fit(
            self.data_storage.X[self.data_storage.labeled_mask],
            self.data_storage.Y[self.data_storage.labeled_mask],
            #  sample_weight=compute_sample_weight(
            #      "balanced",
            #      self.data_storage.Y[self.data_storage.labeled_mask],
            #  ),
        )

    def al_cycle(
        self,
    ) -> None:
        log_it("Started AL Cycle")

        while not self.stopping_criteria.stop_is_reached():
            # try to actively get at least this amount of data, but if there is only less data available that's just fine as well
            if len(self.data_storage.unlabeled_mask) < self.BATCH_SIZE:
                self.BATCH_SIZE = len(self.data_storage.unlabeled_mask)
            if self.BATCH_SIZE == 0:
                # if there is no data left to be labeled we can stop regardless of the stopping criteria
                break

            # potentially the labeling functions have different results now
            self.data_storage.generate_weak_labels(rerun=True)

            self.current_query_indices = (
                self.query_sampling_strategy.get_next_query_indices(self)
            )
            self.current_Y_queries = self.oracle.get_labels(
                self.current_query_indices, self
            )

            self.data_storage.oracle_label_samples(
                self.current_query_indices, self.current_Y_queries
            )

            for callback in self.callbacks.values():
                callback.pre_learning_cycle_hook(self)

            # retrain CLASSIFIER
            self.fit_learner()

            for callback in self.callbacks.values():
                callback.post_learning_cycle_hook(self)
            self.stopping_criteria.update(self)
