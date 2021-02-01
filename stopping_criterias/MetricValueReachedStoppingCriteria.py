from .BaseStoppingCriteria import BaseStoppingCriteria
from ..activeLearner import ActiveLearner


class MetricValueReachedStoppingCriteria(BaseStoppingCriteria):

    current_metric_value: float = 0
    METRIC: str

    def __init(self, STOP_LIMIT: float, METRIC: str) -> None:
        super().__init__(STOP_LIMIT)

        self.METRIC = METRIC

    def stop_is_reached(self) -> bool:
        if self.current_metric_value > self.STOP_LIMIT:
            return True
        else:
            return False

    def update(self, active_learner: ActiveLearner) -> None:
        self.current_metric_value = active_learner.callbacks[self.METRIC].values[-1]
