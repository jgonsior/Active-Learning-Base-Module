import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

if TYPE_CHECKING:
    from active_learning.activeLearner import ActiveLearner
    from .BatchStateEncoding import BatchStateSampling

from active_learning.dataStorage import IndiceMask

from .ImitationLearningBaseQuerySampling import (
    ImitationLearningBaseQuerySampling,
    InputState,
    OutputState,
    PreSampledIndices,
)
from .SingleStateEncoding import SingleStateEncoding


class ImitationLearner(ImitationLearningBaseQuerySampling):
    AMOUNT_OF_PEAKED_OBJECTS: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states: pd.DataFrame = pd.DataFrame(
            data=None,
        )
        self.optimal_policies: pd.DataFrame = pd.DataFrame(
            data=None,
            columns=[
                str(i) + "_true_peaked_normalised_acc"
                for i in range(0, self.AMOUNT_OF_PEAKED_OBJECTS)
            ],
        )

    def save_nn_training_data(self, DATA_PATH):
        self.states.to_csv(
            DATA_PATH + "/states.csv", index=False, header=False, mode="a"
        )
        self.optimal_policies.to_csv(
            DATA_PATH + "/opt_pol.csv", index=False, header=False, mode="a"
        )

    def calculateImitationLearningData(
        self, pre_sampled_X_querie_indices: PreSampledIndices
    ) -> None:
        future_peak_acc = []
        # single thread
        for unlabeled_sample_indices in pre_sampled_X_querie_indices:
            future_peak_acc.append(
                self._future_peak(np.array([unlabeled_sample_indices]))
            )

        self.optimal_policies = self.optimal_policies.append(
            pd.Series(dict(zip(self.optimal_policies.columns, future_peak_acc))),  # type: ignore
            ignore_index=True,
        )

    def encode_input_state(
        self, pre_sampled_X_querie_indices: PreSampledIndices
    ) -> InputState:
        X_state: InputState = super().encode_input_state(pre_sampled_X_querie_indices)
        self.states = self.states.append(
            pd.Series(X_state), ignore_index=True  # type: ignore
        )
        return X_state

    def applyNN(self, X_input_state: InputState) -> OutputState:
        return self.optimal_policies.iloc[-1, :].to_numpy()

    def _future_peak(
        self,
        unlabeled_sample_indices: IndiceMask,
    ) -> float:
        copy_of_classifier = copy.deepcopy(self.learner)

        copy_of_labeled_mask = np.append(
            self.data_storage.labeled_mask, unlabeled_sample_indices, axis=0
        )

        copy_of_classifier.fit(
            self.data_storage.X[copy_of_labeled_mask],
            self.data_storage.Y_merged_final[copy_of_labeled_mask],
        )

        Y_pred_test = copy_of_classifier.predict(self.data_storage.X)
        Y_true = self.data_storage.Y_merged_final

        accuracy_with_that_label = accuracy_score(Y_pred_test, Y_true)

        return accuracy_with_that_label


class TrainImitALSingle(ImitationLearner, SingleStateEncoding):
    pass
