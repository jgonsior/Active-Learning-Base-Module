import abc
from ..activeLearner import ActiveLearner
import numpy as np
from typing import Tuple


class BaseOracle:
    cost: float

    @property
    @abc.abstractmethod
    def identifier(self) -> str:
        pass

    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def has_new_labels(
        self, query_indices: np.ndarray, active_learner: ActiveLearner
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_labels(
        self, query_indices: np.ndarray, active_learner: ActiveLearner
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_oracle_identifier(self) -> str:
        return self.identifier
