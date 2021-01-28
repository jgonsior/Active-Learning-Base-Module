from .BaseStoppingCriteria import BaseStoppingCriteria
from ..active_learner import ActiveLearner


class MetricValueReachedStoppingCriteria(BaseStoppingCriteria):

    current_metric_value = 0

    def __init(self, STOP_LIMIT: float, METRIC: str) -> None:
        super().__init__(STOP_LIMIT)

        self.METRIC = METRIC

    def stop_is_reached(self) -> Bool:
        if self.metric_value > self.STOP_LIMIT:
            return True
        else:
            return False

    def update(self, active_learner: ActiveLearner) -> None:
        self.current_metric_value = active_learner.callback_values[self.METRIC][-1]
