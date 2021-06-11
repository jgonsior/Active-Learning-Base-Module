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

if TYPE_CHECKING:
    from active_learning.dataStorage import (
        DataStorage,
        FeatureList,
        IndiceMask,
        LabelList,
    )


class SyntheticLabelingFunctions(LabelingFunctions):
    identifier: str
    abstain_threshold: float
    clf: Any
    restricted_features: List[int]
    model: str

    def __init__(
        self,
        X: "FeatureList",
        Y: "LabelList",
        AMOUNT_OF_LF_FEATURES: Union[str, float] = "rand",
        LF_CLASSIFIER: str = "rand",
        ABSTAIN_THRESHOLD: Union[str, float] = "rand",
    ):
        self.cost = 0
        self._compute_labeling_function(
            X,
            Y,
            AMOUNT_OF_LF_FEATURES=AMOUNT_OF_LF_FEATURES,
            LF_CLASSIFIER=LF_CLASSIFIER,
            ABSTAIN_THRESHOLD=ABSTAIN_THRESHOLD,
        )

    def labeling_function(
        self, query_indices: "IndiceMask", data_storage: "DataStorage"
    ) -> Tuple["LabelList", LabelConfidence]:
        X = data_storage.X[query_indices]

        if self.model in ["lr", "knn"]:
            X = X[:, self.restricted_features]  # type: ignore
        Y_pred = self.clf.predict(X)
        Y_probas = self.clf.predict_proba(X)

        # return abstain for those samples, for who we are not certain enough
        Y_pred[Y_pred < self.abstain_threshold] = -1

        return Y_pred, Y_probas

    def _compute_labeling_function(
        self,
        X: "FeatureList",
        Y: "LabelList",
        AMOUNT_OF_LF_FEATURES: Union[str, float],
        LF_CLASSIFIER: str,
        ABSTAIN_THRESHOLD: Union[float, str],
    ) -> None:
        """number_of_features: int = random.choices(
            population=range(0, X.shape[1]),
            weights=[i ** 10 for i in reversed(range(0, X.shape[1]))],
            k=1,
        )[0]"""

        if LF_CLASSIFIER == "rand":
            self.model = random.sample(["dt", "lr", "knn"], k=1)[0]
        else:
            self.model = LF_CLASSIFIER

        # focus on a maximum of 3 features
        if AMOUNT_OF_LF_FEATURES == "rand":
            number_of_features: int = random.randint(1, X.shape[1])
        else:
            number_of_features = ceil(AMOUNT_OF_LF_FEATURES * X.shape[1])  # type: ignore

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
        if ABSTAIN_THRESHOLD == "rand":
            self.abstain_threshold = random.random()
        else:
            self.abstain_threshold = ABSTAIN_THRESHOLD  # type: ignore

        self.identifier = (
            "L_"
            + self.model
            + " #"
            + str(number_of_features)
            + ": "
            + str(self.restricted_features)
        )
