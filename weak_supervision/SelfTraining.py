import numpy as np


from typing import Dict, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from active_learning.dataStorage import IndiceMask, LabelList, FeatureList
    from active_learning.dataStorage import DataStorage


from ..learner.standard import Learner

from .BaseWeakSupervision import BaseWeakSupervision


class SelfTraining(BaseWeakSupervision):
    identifier = "U"
    cost = 0

    def __init__(self, CERTAINTY_THRESHOLD, CERTAINTY_RATIO):
        super().__init__()

        self.CERTAINTY_THRESHOLD = CERTAINTY_THRESHOLD
        self.CERTAINTY_RATIO = CERTAINTY_RATIO

        self.identifier += "_" + str(CERTAINTY_THRESHOLD) + "_" + str(CERTAINTY_RATIO)

    def get_labels(self, X: "FeatureList", learner: Learner) -> "LabelList":
        # calculate certainties for all of X_train_unlabeled
        certainties = learner.predict_proba(X)

        Y_pred = learner.predict(X)

        Y_pred[Y_pred > self.CERTAINTY_THRESHOLD] = -1

        return Y_pred
