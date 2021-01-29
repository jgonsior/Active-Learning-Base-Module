# early stop: when stopping criteria says STOP, then we always return that we have no labels left
import abc
from ..active_learner import ActiveLearner


class BaseOracle:
    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def has_new_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> bool:
        if active_learner.stopping_criteria.stop_is_reached():
            return False
        else:
            return True

    @abc.abstractmethod
    def get_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> tuple[list[QueryIndice], list[Label]]:
        pass
