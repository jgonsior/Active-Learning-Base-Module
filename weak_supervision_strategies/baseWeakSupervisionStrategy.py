import abc


class BaseWeakSupervisionStrategy:
    def _init(self, data_storage):
        self.data_storage = data_storage

    @abc.abstractmethod
    def get_weak_requests(self, **kwargs):
        pass
