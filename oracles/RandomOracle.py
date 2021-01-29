import abc
from ..active_learner import ActiveLearner


class BaseOracle:
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
