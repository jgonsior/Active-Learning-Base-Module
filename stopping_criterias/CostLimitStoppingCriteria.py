from .BaseStoppingCriteria import BaseStoppingCriteria
from ..active_learner import ActiveLearner


class CostLimitStoppingCriteria(BaseStoppingCriteria):

    costs_spend = 0

    def stop_is_reached(self) -> Bool:
        if self.costs_spend > self.STOP_LIMIT:
            return True
        else:
            return False

    def update(self, active_learner: ActiveLearner) -> None:
        self.costs_spend += active_learner.data_storage.costs[-1]
