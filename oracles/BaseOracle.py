import abc
from typing import TYPE_CHECKING, Tuple

import numpy as np
if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner


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
        self, query_indices: np.ndarray, active_learner: 'ActiveLearner'
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_labels(
        self, query_indices: np.ndarray, active_learner: 'ActiveLearner'
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_oracle_identifier(self) -> str:
        return self.identifier
