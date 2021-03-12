from .merge_weak_supervision_label_strategies import (
    BaseMergeWeakSupervisionLabelStrategy,
)
import math
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from .weak_supervision import BaseWeakSupervision

# type aliases
IndiceMask = np.ndarray
LabelList = np.ndarray
FeatureList = np.ndarray


class DataStorage:
    unlabeled_mask: IndiceMask
    labeled_mask: IndiceMask
    test_mask: IndiceMask
    weakly_combined_mask: IndiceMask
    weak_supervisions: List[BaseWeakSupervision]

    X: FeatureList
    Y: LabelList
    human_expert_Y: LabelList
    exp_Y: LabelList
    weak_combined_Y: LabelList
    costs_spend: int = 0
    merge_weak_supervision_label_strategy: BaseMergeWeakSupervisionLabelStrategy

    def __init__(
        self,
        df: pd.DataFrame,
        weak_supervisions: List[BaseWeakSupervision],
        merge_weak_supervision_label_strategy: BaseMergeWeakSupervisionLabelStrategy,
        TEST_FRACTION: float = 0.3,
    ) -> None:
        self.TEST_FRACTION: float = TEST_FRACTION
        self.weak_supervisions = weak_supervisions
        self.merge_weak_supervision_label_strategy = (
            merge_weak_supervision_label_strategy
        )

        self.weak_combined_Y = self.Y = self.human_expert_Y = np.ones(len(df)) * -1

        self.X = df.loc[:, df.columns != "label"].to_numpy()  # type: ignore
        self.exp_Y = df["label"].to_numpy().reshape(len(self.X))

        self.label_encoder: LabelEncoder = LabelEncoder()
        # feature normalization
        scaler = RobustScaler()
        self.X = scaler.fit_transform(self.X)

        # scale back to [0,1]
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.label_source = np.full(len(self.exp_Y), "N")

        # check if we are in an experiment setting or are dealing with real, unlabeled data
        if -1 in df["label"]:
            self.Y = self.exp_Y
            # no experiment, we have already some real labels
            self.unlabeled_mask = np.argwhere(pd.isnull(self.Y)).flatten()  # type: ignore
            self.labeled_mask = np.argwhere(~pd.isnull(self.Y)).flatten()  # type: ignore
            self.label_source[self.labeled_mask] = ["G" for _ in self.labeled_mask]
            #  self.Y = Y

            # create test split out of labeled data
            self.test_mask = np.empty(0, dtype=np.int64)

            Y_encoded = self.label_encoder.fit_transform(self.Y[~pd.isnull(self.Y)])
            self.Y = self.Y.astype(np.int64)
            self.Y[self.labeled_mask] = Y_encoded

            self.Y[pd.isnull(self.Y)] = -1
            self.human_expert_Y = self.Y
            self.exp_Y = self.Y

        else:
            # ignore nan as labels
            self.exp_Y = self.label_encoder.fit_transform(
                self.exp_Y[~np.isnan(self.exp_Y)]
            )

            # split into test, train_labeled, train_unlabeled
            # experiment setting apparently

            self.unlabeled_mask = np.arange(
                math.floor(len(self.exp_Y) * self.TEST_FRACTION), len(self.exp_Y)
            )

            # prevent that the first split contains not all labels in the training split, so we just shuffle the data as long as we have every label in their
            while len(np.unique(self.exp_Y[self.unlabeled_mask])) != len(
                self.label_encoder.classes_  # type: ignore
            ):
                new_shuffled_indices = np.random.permutation(len(self.exp_Y))
                self.exp_X = self.X[new_shuffled_indices]
                self.exp_Y = self.exp_Y[new_shuffled_indices]
                self.unlabeled_mask = np.arange(
                    math.floor(len(self.exp_Y) * self.TEST_FRACTION), len(self.exp_Y)
                )
            self.test_mask = np.arange(
                0, math.floor(len(self.exp_Y) * self.TEST_FRACTION)
            )
            self.labeled_mask = np.empty(0, dtype=np.int64)

            """ 
            1. get start_set from X_labeled
            2. if X_unlabeled is None : experiment!
                2.1 if X_test: rest von X_labeled wird X_train_unlabeled
                2.2 if X_test is none: split rest von X_labeled in X_train_unlabeled und X_test
               else (kein experiment):
               X_unlabeled wird X_unlabeled, rest von X_labeled wird X_train_unlabeled_
            
            """
            # separate X_labeled into start_set and labeled _rest
            # check if the minimum amount of labeled data is present in the start set size
            labels_not_in_start_set = set(range(0, len(self.label_encoder.classes_)))  # type: ignore
            all_label_in_start_set = False

            if not all_label_in_start_set:
                #  if len(self.train_labeled_data) == 0:
                #      print("Please specify at least one labeled example of each class")
                #      exit(-1)

                # move more data here from the classes not present
                for label in labels_not_in_start_set:
                    # select a random sample of this labelwhich is NOT yet labeled
                    random_index = np.where(self.exp_Y[self.unlabeled_mask] == label)[
                        0
                    ][0]

                    # the random_index before is an index on Y[unlabeled_QueryIndices], and therefore NOT the same as an index on purely Y
                    # therefore it needs to be converted first
                    random_index = self.unlabeled_mask[random_index]

                    self.label_samples(
                        np.array([random_index]),
                        np.array([label]),
                        "S",
                    )

    def unlabel_samples(self, query_indices: IndiceMask) -> None:

        self.unlabeled_mask = np.append(self.unlabeled_mask, query_indices, axis=0)

        for sample in query_indices:
            self.labeled_mask = self.labeled_mask[self.labeled_mask != sample]

        self.human_expert_Y[query_indices] = -1
        self.Y[query_indices] = -1

    def update_samples(self, query_indices: IndiceMask, Y_query: LabelList) -> None:
        self.human_expert_Y[query_indices] = Y_query
        self.Y[query_indices] = Y_query

    def label_samples(
        self, query_indices: IndiceMask, Y_queries: LabelList, label_source: str
    ) -> None:
        self.labeled_mask = np.append(self.labeled_mask, query_indices, axis=0)  # type: ignore

        for element in query_indices:
            self.unlabeled_mask = self.unlabeled_mask[self.unlabeled_mask != element]

        self.label_source[query_indices] = label_source
        self.human_expert_Y[query_indices] = Y_queries
        self.Y[query_indices] = Y_queries

        # todo later: eventuell basierend auf Daten mit dazu rechnen, dass Label verschiedene Kosten haben (Annotationsdauer in Experimenten etc)
        self.costs_spend += len(query_indices)

    def get_experiment_labels(self, query_indice: IndiceMask) -> LabelList:
        return self.exp_Y[query_indice]

    def generate_weak_labels(self) -> None:
        # store in weak_combined_Y
        # merge together with human_expert_Y into Y
        ws_labels_list: List[LabelList] = []
        for weak_supervision in self.weak_supervisions:
            ws_labels_list.append(
                weak_supervision.get_labels(self.X[self.unlabeled_mask])
            )

        # magic
        self.weak_combined_Y[
            self.unlabeled_mask
        ] = self.merge_weak_supervision_label_strategy.merge(ws_labels_list)

        self.Y[self.unlabeled_mask] = self.weak_combined_Y[self.unlabeled_mask]
