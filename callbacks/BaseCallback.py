import abc
from ..activeLearner import ActiveLearner
from typing import List, Any


class BaseCallback:
    values: List[Any] = []

    @abc.abstractmethod
    def pre_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass

    @abc.abstractmethod
    def post_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass
