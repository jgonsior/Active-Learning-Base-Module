import abc
from ..active_learner import ActiveLearner
import numpy as np
from typing import Tuple


class BaseOracle:
    @property
    @abc.abstractmethod
    def identifier(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def cost(self) -> float:
        pass

    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def has_new_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_labels(
        self, query_indices: np.ndarray[np.int64], active_learner: ActiveLearner
    ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
        pass

    def get_oracle_identifier(self) -> str:
        return self.identifier
