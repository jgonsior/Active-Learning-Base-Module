import abc
from ..active_learner import ActiveLearner
from .BaseOracle import BaseOracle


class FakeExperimentOracle(BaseOracle):
    identifier = "E"
    cost = 1

    def has_new_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> bool:
        if active_learner.stopping_criteria.stop_is_reached():
            return False
        else:
            return True

    def get_labels(
        self, query_indices: list[QueryIndice], active_learner: ActiveLearner
    ) -> tuple[list[QueryIndice], list[Label]]:
        return query_indices, active_learner.data_storage.get_experiment_labels(
            query_indices
        )
