import abc


from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from active_learning.dataStorage import IndiceMask, LabelList
    from active_learning.dataStorage import DataStorage


class BaseWeakSupervision(abc.ABC):
    @abc.abstractmethod
    def get_labels(
        self, query_indices: "IndiceMask", data_storage: "DataStorage"
    ) -> "LabelList":
        pass
