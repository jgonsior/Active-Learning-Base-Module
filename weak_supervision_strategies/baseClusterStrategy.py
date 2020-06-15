import abc


class BaseClusterStrategy:
    @abc.abstractmethod
    def get_weak_requests(self, **kwargs):
        pass
