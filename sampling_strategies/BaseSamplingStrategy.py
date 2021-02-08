import abc
from active_learning.dataStorage import DataStorage, IndiceMask
from ..learner.standard import Learner


class BaseSamplingStrategy:
    def __init__(self):
        self.values = []

    @abc.abstractmethod
    def what_to_label_next(
        self, NR_QUERIES_PER_ITERATION: int, learner: Learner, data_storage: DataStorage
    ) -> IndiceMask:
        pass
