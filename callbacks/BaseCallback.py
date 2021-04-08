import abc
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner


class BaseCallback(abc.ABC):
    values: List[Any] = []

    @abc.abstractmethod
    def pre_learning_cycle_hook(self, active_learner: "ActiveLearner") -> None:
        pass

    @abc.abstractmethod
    def post_learning_cycle_hook(self, active_learner: "ActiveLearner") -> None:
        pass
