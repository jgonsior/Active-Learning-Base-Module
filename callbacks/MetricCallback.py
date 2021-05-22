from sklearn.metrics import accuracy_score, f1_score
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner

from .BaseCallback import BaseCallback


class MetricCallback(BaseCallback):
    def __init__(
        self, metric_function: Callable[["ActiveLearner"], List[float]]
    ) -> None:
        self.metric_function = metric_function
        self.values = []

    def pre_learning_cycle_hook(self, active_learner: "ActiveLearner") -> None:
        pass

    def post_learning_cycle_hook(self, active_learner: "ActiveLearner") -> None:
        self.values.append(self.metric_function(active_learner))


def test_f1_metric(active_learner: "ActiveLearner") -> List[float]:
    Y_true = active_learner.data_storage.exp_Y[active_learner.data_storage.test_mask]
    Y_pred = active_learner.learner.predict(
        active_learner.data_storage.X[active_learner.data_storage.test_mask]
    )
    return f1_score(Y_true, Y_pred, average="weighted", zero_division=0)  # type: ignore


def test_acc_metric(active_learner: "ActiveLearner") -> List[float]:
    Y_true = active_learner.data_storage.exp_Y[active_learner.data_storage.test_mask]
    Y_pred = active_learner.learner.predict(
        active_learner.data_storage.X[active_learner.data_storage.test_mask]
    )

    return accuracy_score(Y_true, Y_pred)  # type: ignore


def get_single_al_run_stats_table_header():
    return "Iteration: {:>3} {:>6} {:>6} {:>6} {:>6} {:>6} {:>3}".format(
        "I", "L", "U", "Q", "Te", "Tr", "R"
    )


def get_single_al_run_stats_row(
    i, amount_of_labeled, amount_of_unlabeled, metrics_per_al_cycle, index=-1
):
    if amount_of_labeled == None:
        amount_of_labeled = 0
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_labeled += query_length

        amount_of_unlabeled = 2889
        for query_length in metrics_per_al_cycle["query_length"][:index]:
            amount_of_unlabeled -= query_length

    return "Iteration: {:3,d} {:6,d} {:6,d} {:6,d} {:6.1%} {:6.1%} {:>3}".format(
        i,
        amount_of_labeled,
        amount_of_unlabeled,
        metrics_per_al_cycle["query_length"][index],
        metrics_per_al_cycle["test_f1"][index],
        metrics_per_al_cycle["train_f1"][index],
        metrics_per_al_cycle["source"][index],
    )
