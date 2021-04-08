from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner
from .BaseStoppingCriteria import BaseStoppingCriteria


class CostLimitStoppingCriteria(BaseStoppingCriteria):

    costs_spend: float = 0

    def stop_is_reached(self) -> bool:
        if self.costs_spend > self.STOP_LIMIT:
            return True
        else:
            return False

    def update(self, active_learner: "ActiveLearner") -> None:
        self.costs_spend += active_learner.data_storage.costs_spend[-1]
