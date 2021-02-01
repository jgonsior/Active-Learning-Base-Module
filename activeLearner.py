import abc
import numpy as np
from typing import List

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from .callbacks.BaseCallback import BaseCallback
from .dataStorage import DataStorage
from .logger.logger import log_it
from .oracles.BaseOracle import BaseOracle
from .oracles.LabeledStartSetOracle import LabeledStartSetOracle
from .sampling_strategies import BaseSamplingStrategy
from .stopping_criterias.BaseStoppingCriteria import BaseStoppingCriteria
from typing import NewType, Union, TypeVar, Type, Dict

Learner = TypeVar(
    "Learner",
    MLPClassifier,
    SVC,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GaussianNB,
)


class ActiveLearner:
    def __init__(
        self,
        sampling_strategy: BaseSamplingStrategy,
        data_storage: DataStorage,
        oracles: List[BaseOracle],
        learner: Learner,
        callbacks: Dict[str, BaseCallback],
        stopping_criteria: BaseStoppingCriteria,
        NR_QUERIES_PER_ITERATION,
    ) -> None:

        self.sampling_strategy = sampling_strategy
        self.data_storage = data_storage
        self.learner = learner
        self.oracles = oracles
        self.callbacks = callbacks
        self.stopping_criteria = stopping_criteria
        self.NR_QUERIES_PER_ITERATION = NR_QUERIES_PER_ITERATION

        # fake iteration zero
        self.current_query_indices = self.data_storage.labeled_mask

        self.current_Y_query = self.data_storage.Y[self.data_storage.labeled_mask]

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
        while not self.stopping_criteria.stop_is_reached():
            # try to actively get at least this amount of data, but if there is only less data available that's just fine as well
            if len(self.data_storage.unlabeled_mask) < self.NR_QUERIES_PER_ITERATION:
                self.NR_QUERIES_PER_ITERATION = len(self.data_storage.unlabeled_mask)
            if self.NR_QUERIES_PER_ITERATION == 0:
                # if there is no data left to be labeled we can stop regardless of the stopping criteria
                break

            Y_query: np.ndarray = np.empty(0, dtype=np.int64)

            query_indices = self.sampling_strategy.what_to_label_next(
                self.NR_QUERIES_PER_ITERATION, self.learner, self.data_storage
            )

            oracle: Union[BaseOracle, None] = None

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

            if oracle is None:
                log_it(
                    "No oracle available, exiting, implement logic for that corner case later"
                )
                exit(-1)

            self.data_storage.label_samples(
                query_indices,
                Y_query,
                self.current_oracle.get_oracle_identifier(),
                oracle.cost,
            )

            for callback in self.callbacks.values():
                callback.pre_learning_cycle_hook(self)

            # retrain CLASSIFIER
            self.fit_learner()

            for callback in self.callbacks.values():
                callback.post_learning_cycle_hook(self)
            self.stopping_criteria.update(self)
