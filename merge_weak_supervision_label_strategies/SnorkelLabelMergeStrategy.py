from snorkel.labeling.model import LabelModel
from typing import List, TYPE_CHECKING
import numpy as np
from .BaseMergeWeakSupervisionLabelStrategy import BaseMergeWeakSupervisionLabelStrategy

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from active_learning.dataStorage import IndiceMask, LabelList


class SnorkelLabelMergeStrategy(BaseMergeWeakSupervisionLabelStrategy):
    def __init__(self, cardinality: int, random_seed: int):
        self.cardinality = cardinality
        self.random_seed = random_seed

    def merge(self, ws_labels_list: np.ndarray) -> "LabelList":
        label_model = LabelModel(cardinality=self.cardinality, verbose=True)
        label_model.fit(
            L_train=ws_labels_list, n_epochs=500, log_freq=100, seed=self.random_seed
        )
        return label_model.predict(ws_labels_list)
