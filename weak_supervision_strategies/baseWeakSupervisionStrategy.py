import abc


class BaseWeakSupervisionStrategy:
    @abc.abstractmethod
    def get_weak_requests(self, **kwargs):
        pass
