from typing import Dict, List, Union

from .callbacks.BaseCallback import BaseCallback
from .dataStorage import DataStorage, IndiceMask, LabelList
from .learner.standard import Learner
from .logger.logger import log_it
from .oracles.BaseOracle import BaseOracle
from .oracles.LabeledStartSetOracle import LabeledStartSetOracle
from .sampling_strategies.BaseSamplingStrategy import BaseSamplingStrategy
from .stopping_criterias.BaseStoppingCriteria import BaseStoppingCriteria


class ActiveLearner:
    def __init__(
        self,
        sampling_strategy: BaseSamplingStrategy,
        data_storage: DataStorage,
        oracles: List[BaseOracle],
        learner: Learner,
        callbacks: Dict[str, BaseCallback],
        stopping_criteria: BaseStoppingCriteria,
        BATCH_SIZE: int,
    ) -> None:

        self.sampling_strategy: BaseSamplingStrategy = sampling_strategy
        self.data_storage: DataStorage = data_storage
        self.learner: Learner = learner
        self.oracles: List[BaseOracle] = oracles
        self.callbacks: Dict[str, BaseCallback] = callbacks
        self.stopping_criteria: BaseStoppingCriteria = stopping_criteria
        self.BATCH_SIZE: int = BATCH_SIZE

        # fake iteration zero
        self.current_query_indices: IndiceMask = self.data_storage.labeled_mask

        self.current_Y_query: LabelList = self.data_storage.Y[
            self.data_storage.labeled_mask
        ]

        self.current_oracle: BaseOracle = LabeledStartSetOracle()

        for callback in self.callbacks.values():
            callback.pre_learning_cycle_hook(self)

        # retrain CLASSIFIER
        self.fit_learner()

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
        print(self.oracles)
        while not self.stopping_criteria.stop_is_reached():
            # try to actively get at least this amount of data, but if there is only less data available that's just fine as well
            if len(self.data_storage.unlabeled_mask) < self.BATCH_SIZE:
                self.BATCH_SIZE = len(self.data_storage.unlabeled_mask)
            if self.BATCH_SIZE == 0:
                # if there is no data left to be labeled we can stop regardless of the stopping criteria
                break

            query_indices = self.sampling_strategy.what_to_label_next(
                self.BATCH_SIZE, self.learner, self.data_storage
            )

            self.current_oracle: Union[BaseOracle, None] = None

            # @todo what about weak supervision and multiple oracle?
            # should MY net decide, which oracle to trust, or is my oracle still only there to decide, which label to take,
            # and then I've got another NN next to it, which decides, which result to take?
            # or does my NN also specifies which oracle to ask?!?!
            for oracle in self.oracles:
                # decide somehow which oracle to ask

                if oracle.has_new_labels(query_indices, self):
                    (
                        self.current_query_indices,
                        self.current_Y_query,
                    ) = oracle.get_labels(query_indices, self)
                    self.current_oracle = oracle
                    break

            if self.current_oracle is None:
                log_it(
                    "No oracle available, exiting, implement logic for that corner case later"
                )
                exit(-1)

            # even though oracle, query_indices and Y_query are local variables for this cycle we store them as object variables to let f.e. callbacks/metric access them
            self.data_storage.label_samples(
                self.current_query_indices,
                self.current_Y_query,
                self.current_oracle.get_oracle_identifier(),
                self.current_oracle.cost,
            )

            for callback in self.callbacks.values():
                callback.pre_learning_cycle_hook(self)

            # retrain CLASSIFIER
            self.fit_learner()

            for callback in self.callbacks.values():
                callback.post_learning_cycle_hook(self)
            self.stopping_criteria.update(self)
