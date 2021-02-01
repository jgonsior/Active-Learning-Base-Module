import abc
from ..activeLearner import ActiveLearner


class BaseStoppingCriteria:
    def __init__(self, STOP_LIMIT: float):
        self.STOP_LIMIT = STOP_LIMIT

    @abc.abstractmethod
    def stop_is_reached(self) -> bool:
        pass

    @abc.abstractmethod
    def update(self, active_learner: ActiveLearner) -> None:
        pass
