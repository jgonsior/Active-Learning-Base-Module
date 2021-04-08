from sklearn.metrics import accuracy_score, f1_score
from typing import TYPE_CHECKING, Callable, List

from active_learning.logger.logger import log_it

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner

from .BaseCallback import BaseCallback


class PrintLoggingStatisticsCallback(BaseCallback):
    ITERATION_COUNTER: int = 0

    def pre_learning_cycle_hook(self, active_learner: "ActiveLearner") -> None:
        pass

    # ensure that the metric callbacks are called first (just specify them first in the dict)
    def post_learning_cycle_hook(self, active_learner: "ActiveLearner") -> None:
        self.ITERATION_COUNTER += 1
        log_it(
            "{:>2} {:>6} {:>6} {:.2%} {:.2%}".format(
                self.ITERATION_COUNTER,
                len(active_learner.data_storage.labeled_mask),
                len(active_learner.data_storage.unlabeled_mask),
                active_learner.callbacks["acc_test"].values[-1],
                active_learner.callbacks["f1_test"].values[-1],
            )
        )
        pass
