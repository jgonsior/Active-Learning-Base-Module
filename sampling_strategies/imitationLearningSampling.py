from train_lstm import AMOUNT_OF_PEAKED_OBJECTS
from typing import List
from active_learning.learner.standard import Learner
from active_learning.activeLearner import ActiveLearner
from active_learning.dataStorage import FeatureList, IndiceMask, LabelList
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .learnedBaseSampling import LearnedBaseSampling, State



class ImitationLearner(LearnedBaseSampling):
    AMOUNT_OF_PEAKED_OBJECTS: int

    def __init__(self, AMOUNT_OF_PEAKED_OBJECTS: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AMOUNT_OF_PEAKED_OBJECTS = AMOUNT_OF_PEAKED_OBJECTS
        self.states:pd.DataFrame = pd.DataFrame(
            data=None,
        )
        self.optimal_policies:pd.DataFrame = pd.DataFrame(
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

    def get_X_query_index(self) -> IndiceMask:
        future_peak_acc = []

        possible_samples_indices = self.pre_sample_potential_X_queries(
            self.AMOUNT_OF_PEAKED_OBJECTS
        )

        future_peak_acc = []
        # single thread
        for unlabeled_sample_indices in possible_samples_indices:
            future_peak_acc.append(
                self._future_peak(
                    unlabeled_sample_indices
                )
            )

        # parallelisieren
        #  with parallel_backend("loky", n_jobs=self.N_JOBS):
        #      future_peak_acc = Parallel()(
        #          delayed(self._future_peak)(
        #              unlabeled_sample_index,
        #              self.weak_supervision_label_sources,
        #              self.data_storage,
        #              self.clf,
        #              self.MAX_AMOUNT_OF_WS_PEAKS,
        #          )
        #          for unlabeled_sample_index in possible_samples_indices
        #      )
        #
       
        self.optimal_policies = self.optimal_policies.append(
            pd.Series(dict(zip(self.optimal_policies.columns, future_peak_acc))), # type: ignore
            ignore_index=True,
        )
        return possible_samples_indices

    def calculate_next_query_indices_post_hook(self, X_state: State) -> None:
        self.states = self.states.append(
            pd.Series(X_state), # type: ignore
            ignore_index=True
            #  pd.Series(dict(zip(self.states.columns, X_state))), ignore_index=True,
        )

    def get_sorting(self, _)->List[float]:
        return [i for i in self.optimal_policies.iloc[-1, :]]

    def _future_peak(
        self,
        unlabeled_sample_indices: IndiceMask,
    ):
        copy_of_learner = copy.deepcopy(self.learner)

        copy_of_labeled_mask = np.append(
            self.data_storage.labeled_mask, unlabeled_sample_indices, axis=0
        )

        copy_of_learner.fit(
            self.data_storage.X[copy_of_labeled_mask], self.data_storage.Y[copy_of_labeled_mask]
        )

        Y_pred_test = copy_of_learner.predict(self.data_storage.X)
        Y_true = self.data_storage.Y

        accuracy_with_that_label = accuracy_score(Y_pred_test, Y_true)
        #  accuracy_with_that_label = f1_score(
        #      Y_pred_test, Y_true, average="weighted", zero_division=0
        #  )
        #  print(
        #      "Testing out : {}, test acc: {}".format(
        #          unlabeled_sample_index, accuracy_with_that_label
        #      )
        #  )
        return accuracy_with_that_label
