from active_learning.dataStorage import IndiceMask
import copy
from ALiPy.alipy.metrics.performance import accuracy_score
import numpy as np

from .query_sampling_strategies.ImitationLearningBaseQuerySampler import (
    InputState,
    OutputState,
    PreSampledIndices,
)
from .query_sampling_strategies.TrainedImitALQuerySampler import (
    TrainedImitALBatchSampler,
    TrainedImitALSampler,
    TrainedImitALSingleSampler,
)


class ALiPY_Optimal_Query_Strategy:
    trained_imitAL_sampler: TrainedImitALSampler
    X: np.ndarray
    Y: np.ndarray
    AMOUNT_OF_PEAKED_OBJECTS: int = 20

    def __init__(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        self.X = X
        self.Y = Y

    def select(self, labeled_index, unlabeled_index, model, batch_size=1, **kwargs):
        if self.AMOUNT_OF_PEAKED_OBJECTS > len(unlabeled_index):
            replace = True
        else:
            replace = False
        pre_sampled_X_querie_indices: PreSampledIndices = np.random.choice(
            unlabeled_index,
            size=self.AMOUNT_OF_PEAKED_OBJECTS,
            replace=replace,
        )

        future_peak_acc = []
        # single thread
        for unlabeled_sample_indices in pre_sampled_X_querie_indices:
            future_peak_acc.append(
                self._future_peak(
                    model,
                    labeled_index,
                    np.array([unlabeled_sample_indices]),
                )
            )

        zero_to_one_values_and_index = set(
            zip(future_peak_acc, pre_sampled_X_querie_indices)
        )

        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return [v for _, v in ordered_list_of_possible_sample_indices[:batch_size]]

    def _future_peak(
        self,
        model,
        labeled_index,
        unlabeled_sample_indices: IndiceMask,
    ) -> float:

        copy_of_classifier = copy.deepcopy(model)

        copy_of_labeled_mask = np.append(
            labeled_index, unlabeled_sample_indices, axis=0
        )

        copy_of_classifier.fit(
            self.X[copy_of_labeled_mask],
            self.Y[copy_of_labeled_mask],
        )

        Y_pred_test = copy_of_classifier.predict(self.X)
        Y_true = self.Y

        accuracy_with_that_label = accuracy_score(Y_pred_test, Y_true)

        return accuracy_with_that_label
