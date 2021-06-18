import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Any, List, Tuple, Union
from typing import List, TYPE_CHECKING
from math import ceil
from active_learning.logger.logger import log_it
from .LabelingFunctions import (
    LabelConfidence,
    LabelingFunctions,
)
from scipy.stats import loguniform

if TYPE_CHECKING:
    from active_learning.dataStorage import (
        DataStorage,
        FeatureList,
        IndiceMask,
        LabelList,
    )
from active_learning.learner.standard import Learner


class SyntheticLabelingFunctions(LabelingFunctions):
    identifier: str
    ABSTAIN_THRESHOLD: float
    LF_CLASSIFIER_NAME: str
    AMOUNT_OF_LF_FEATURESSS: int
    restricted_features: List[int]
    RANDOM_SEED: int
    model: Learner

    def __init__(self, X: "FeatureList", Y: "LabelList", RANDOM_SEED: int):
        self.cost = 0

        self.RANDOM_SEED = RANDOM_SEED
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self._compute_labeling_function(
            X,
            Y,
        )

    def labeling_function(
        self, query_indices: "IndiceMask", data_storage: "DataStorage"
    ) -> Tuple["LabelList", LabelConfidence]:
        X = data_storage.X[query_indices]

        if self.LF_CLASSIFIER_NAME in ["lr", "knn"]:
            X = X[:, self.restricted_features]  # type: ignore
        Y_pred = self.model.predict(X)
        Y_probas = self.model.predict_proba(X)

        # return abstain for those samples, for who we are not certain enough
        Y_pred[Y_pred < self.ABSTAIN_THRESHOLD] = -1

        return Y_pred, Y_probas

    def _compute_labeling_function(
        self,
        X: "FeatureList",
        Y: "LabelList",
    ) -> None:
        """number_of_features: int = random.choices(
            population=range(1, min(10, X.shape[1])),
            weights=[i ** 10 for i in reversed(range(1, min(10, X.shape[1])))],
            k=1,
        )[0]"""
        self.LF_CLASSIFIER_NAME = random.sample(["dt", "lr", "knn"], k=1)[0]

        number_of_features: int = ceil(
            loguniform.rvs(a=0.01, b=0.5, size=1, random_state=self.RANDOM_SEED)
            * min(10, X.shape[1])
        )

        self.restricted_features = random.sample(
            [i for i in range(0, X.shape[1])], k=number_of_features
        )

        self.AMOUNT_OF_LF_FEATURESSS = number_of_features

        log_it(
            "LF: "
            + self.LF_CLASSIFIER_NAME
            + " #"
            + str(number_of_features)
            + ": "
            + str(self.restricted_features)
        )
        if self.LF_CLASSIFIER_NAME == "dt":
            # train a shallow decision tree focussing only on those two features
            clf = DecisionTreeClassifier(max_depth=number_of_features)
            clf.fit(X, Y)
        elif self.LF_CLASSIFIER_NAME == "lr":
            clf = LogisticRegression()

            # only use the two specified features
            clf.fit(X[:, self.restricted_features], Y)  # type: ignore
        elif self.LF_CLASSIFIER_NAME == "knn":
            clf = KNeighborsClassifier(algorithm="kd_tree")
            clf.fit(X[:, self.restricted_features], Y)  # type: ignore

        self.model = clf  # type: ignore

        # calculate a random threshold under which the lf abstains
        self.ABSTAIN_THRESHOLD = random.random()

        self.identifier = (
            "L_"
            + self.LF_CLASSIFIER_NAME
            + " #"
            + str(number_of_features)
            + ": "
            + str(self.restricted_features)
        )
