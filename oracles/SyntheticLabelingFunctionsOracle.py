from active_learning.oracles.LabelingFunctionsOracle import (
    LabelConfidence,
    LabeleingFunctionsOracle,
)
from active_learning.logger.logger import log_it
import random
from active_learning.dataStorage import DataStorage, FeatureList, IndiceMask, LabelList
from typing import Any, List, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class SyntheticLabelingFunctionsOracle(LabeleingFunctionsOracle):
    identifier: str
    abstain_threshold: float
    clf: Any
    restricted_features: List[int]
    model: str

    def __init__(self, X: FeatureList, Y: LabelList, error_factor: float = 0.9):
        self.cost = 0
        self._compute_labeling_function(X, Y, error_factor=error_factor)

    def labeling_function(
        self, query_indices: IndiceMask, data_storage: DataStorage
    ) -> Tuple[LabelList, LabelConfidence]:
        X = data_storage.X[query_indices]

        if self.model in ["lr", "knn"]:
            X = X[:, self.restricted_features]  # type: ignore
        Y_pred = self.clf.predict(X)
        Y_probas = self.clf.predict_proba(X)

        # return abstain for those samples, for who we are not certain enough
        Y_pred[Y_pred < self.abstain_threshold] = -1

        return Y_pred, Y_probas

    def _compute_labeling_function(
        self, X: FeatureList, Y: LabelList, error_factor: float
    ) -> None:
        """number_of_features: int = random.choices(
            population=range(0, X.shape[1]),
            weights=[i ** 10 for i in reversed(range(0, X.shape[1]))],
            k=1,
        )[0]"""

        self.model = random.sample(["dt", "lr", "knn"], k=1)[0]

        # focus on a maximum of 3 features
        number_of_features: int = random.randint(1, min(3, X.shape[1]))

        self.restricted_features = random.sample(
            [i for i in range(0, X.shape[1])], k=number_of_features
        )
        log_it(
            "LF: "
            + self.model
            + " #"
            + str(number_of_features)
            + ": "
            + str(self.restricted_features)
        )
        if self.model == "dt":
            # train a shallow decision tree focussing only on those two features
            clf = DecisionTreeClassifier(max_depth=number_of_features)
            clf.fit(X, Y)
        elif self.model == "lr":
            clf = LogisticRegression()

            # only use the two specified features
            clf.fit(X[:, self.restricted_features], Y)  # type: ignore
        elif self.model == "knn":
            clf = KNeighborsClassifier(algorithm="kd_tree")
            clf.fit(X[:, self.restricted_features], Y)  # type: ignore

        self.clf = clf  # type: ignore
        # calculate a random threshold under which the lf abstains
        self.abstain_threshold = random.random()

        self.identifier = (
            "L_"
            + self.model
            + " #"
            + str(number_of_features)
            + ": "
            + str(self.restricted_features)
        )
