import random
from active_learning.dataStorage import FeatureList, IndiceMask, LabelList
from typing import Any, List, Sequence, TYPE_CHECKING, Callable, Tuple
from sklearn.tree import DecisionTreeClassifier
import numpy as np

if TYPE_CHECKING:
    from ..activeLearner import ActiveLearner

from .BaseOracle import BaseOracle


class LabeleingFunctionsOracle(BaseOracle):
    identifier = "F"

    def __init__(
        self,
        labeling_function: Callable[[FeatureList], Tuple[LabelList, np.ndarray]],
        cost: float,
    ):
        super().__init__()

        self.labeling_function = labeling_function
        self.cost = cost

    def has_new_labels(
        self, query_indices: IndiceMask, active_learner: "ActiveLearner"
    ) -> bool:
        self.potentially_labeled_query_indices, self.Y_lf = self.labeling_function(
            active_learner.data_storage.X[active_learner.data_storage.unlabeled_mask]
        )
        if len(self.potentially_labeled_query_indices) > 0:
            return False
        else:
            return True

    def get_labels(
        self, query_indices: IndiceMask, active_learner: "ActiveLearner"
    ) -> Tuple[IndiceMask, LabelList]:
        return self.potentially_labeled_query_indices, self.Y_lf


def compute_labeling_function(
    X: FeatureList, Y: LabelList, error_factor: float = 0.9
) -> Callable[[FeatureList], Tuple[LabelList, np.ndarray]]:
    """number_of_features: int = random.choices(
        population=range(0, X.shape[1]),
        weights=[i ** 10 for i in reversed(range(0, X.shape[1]))],
        k=1,
    )[0]"""

    # focus on a maximum of 3 features
    number_of_features: int = random.randint(1, min(3, X.shape[1]))

    # train a shallow decision tree only on those two features
    dt = DecisionTreeClassifier(max_depth=number_of_features)
    dt.fit(X, Y)

    # calculate a random threshold under which the lf abstains
    abstain_threshold: float = random.random()

    def _lf(X: FeatureList) -> Tuple[LabelList, np.ndarray]:
        Y_pred = dt.predict(X)
        Y_probas = dt.predict_log_proba(X)

        Y_pred[Y_pred < abstain_threshold] = -1
        return Y_pred, Y_probas

    return _lf


def get_n_labeling_functions(
    amount: int,
    X: FeatureList,
    Y: LabelList,
    error_factor: float = 0.9,
    cost: float = 0,
) -> List[BaseOracle]:
    return [
        LabeleingFunctionsOracle(
            compute_labeling_function(X, Y, error_factor), cost=cost
        )
        for _ in range(0, amount)
    ]
