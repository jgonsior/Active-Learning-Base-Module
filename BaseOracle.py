from abc import ABC, abstractmethod


def Oracle(ABC):
    @abstractmethod
    def get_next_indices(self, data_storage):
