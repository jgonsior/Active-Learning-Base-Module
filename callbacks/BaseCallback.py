import abc
from ..active_learner import ActiveLearner


class BaseCallback:
    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def pre_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass

    @abc.abstractmethod
    def post_learning_cycle_hook(self, active_learner: ActiveLearner) -> None:
        pass
