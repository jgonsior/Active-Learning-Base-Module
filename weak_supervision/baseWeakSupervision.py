import abc


class BaseWeakSupervision:
    def _init(self, data_storage, **THRESHOLDS):
        self.data_storage = data_storage

        for THRESHOLD in THRESHOLDS:
            setattr(self, k, v)

    @abc.abstractmethod
    def get_labeled_samples(self, **kwargs):
        pass
