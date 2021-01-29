import abc
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle
import random


class RandomOracle(BaseOracle):
    identifier = "R"
    cost = -1

    def has_new_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> bool:
        return True

    def get_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> tuple[list[QueryIndice], list[Label]]:
        return query_indices, [
            random.randint(0, len(active_learner.data_storage.label_encoder.classes_))
            for _ in range(0, len(query_indices))
        ]
