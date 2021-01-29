import abc
from ..active_learner import ActiveLearner


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
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> bool:
        pass

    @abc.abstractmethod
    def get_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> tuple[list[QueryIndice], list[Label]]:
        pass

    def get_oracle_identifier(self) -> str:
        return self.identifier
