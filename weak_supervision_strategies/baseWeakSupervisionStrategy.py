import abc


class BaseWeakSupervisionStrategy:
    def _init(self, data_storage, **WS_THRESHOLDS):
        self.data_storage = data_storage

        for WS_THRESHOLD in WS_THRESHOLDS:
            setattr(self, k, v)

    @abc.abstractmethod
    def get_weak_requests(self, **kwargs):
        pass
