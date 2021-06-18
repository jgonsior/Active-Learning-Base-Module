import math
import numpy as np
import pandas as pd
from distutils.command.config import config
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from typing import List, Optional

from active_learning.logger.logger import log_it
from .merge_weak_supervision_label_strategies import (
    BaseMergeWeakSupervisionLabelStrategy,
)
from .weak_supervision import BaseWeakSupervision
from .learner.standard import Learner

# type aliases
IndiceMask = np.ndarray
LabelList = np.ndarray
FeatureList = np.ndarray


class DataStorage:
    unlabeled_mask: IndiceMask
    labeled_mask: IndiceMask
    test_mask: IndiceMask
    weakly_combined_mask: IndiceMask
    weak_supervisions: List[BaseWeakSupervision] = []
    ws_labels_list: np.ndarray

    X: FeatureList
    Y_merged_final: LabelList  # the final label, merged from human_expert_Y and weak_combined_Y -> this is the stuff which trains the AL-model
    human_expert_Y: LabelList  # those who come from the human oracle
    true_Y: LabelList  # those who are known from the beginning in an experiment setting
    weak_combined_Y: LabelList  # the merged labels from all the weak supervision sources
    costs_spend: int = 0
    merge_weak_supervision_label_strategy: BaseMergeWeakSupervisionLabelStrategy

    def __init__(
        self,
        df: pd.DataFrame,
        TEST_FRACTION: float = 0.3,
    ) -> None:
        self.TEST_FRACTION: float = TEST_FRACTION

        self.weak_combined_Y = self.Y_merged_final = self.human_expert_Y = (
            np.ones(len(df)) * -1
        )

        self.X = df.loc[:, df.columns != "label"].to_numpy()  # type: ignore
        self.true_Y = df["label"].to_numpy().reshape(len(self.X))

        self.label_encoder: LabelEncoder = LabelEncoder()
        # feature normalization
        scaler = RobustScaler()
        self.X = scaler.fit_transform(self.X)

        # scale back to [0,1]
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.label_source = np.full(len(self.true_Y), "N")

        # check if we are in an experiment setting or are dealing with real, unlabeled data
        if -1 in df["label"]:
            self.Y_merged_final = self.true_Y
            # no experiment, we have already some real labels
            self.unlabeled_mask = np.argwhere(pd.isnull(self.Y_merged_final)).flatten()  # type: ignore
            self.labeled_mask = np.argwhere(~pd.isnull(self.Y_merged_final)).flatten()  # type: ignore
            self.label_source[self.labeled_mask] = ["G" for _ in self.labeled_mask]
            #  self.Y = Y

            # create test split out of labeled data
            self.test_mask = np.empty(0, dtype=np.int64)

            Y_encoded = self.label_encoder.fit_transform(
                self.Y_merged_final[~pd.isnull(self.Y_merged_final)]
            )
            self.Y_merged_final = self.Y_merged_final.astype(np.int64)
            self.Y_merged_final[self.labeled_mask] = Y_encoded

            self.Y_merged_final[pd.isnull(self.Y_merged_final)] = -1
            self.human_expert_Y = self.Y_merged_final
            self.true_Y = self.Y_merged_final

        else:
            # ignore nan as labels
            self.true_Y = self.label_encoder.fit_transform(
                self.true_Y[~np.isnan(self.true_Y)]
            )

            # split into test, train_labeled, train_unlabeled
            # experiment setting apparently

            self.unlabeled_mask = np.arange(
                math.floor(len(self.true_Y) * self.TEST_FRACTION), len(self.true_Y)
            )

            # prevent that the first split contains not all labels in the training split, so we just shuffle the data as long as we have every label in their
            while len(np.unique(self.true_Y[self.unlabeled_mask])) != len(
                self.label_encoder.classes_  # type: ignore
            ):
                new_shuffled_indices = np.random.permutation(len(self.true_Y))
                self.exp_X = self.X[new_shuffled_indices]
                self.true_Y = self.true_Y[new_shuffled_indices]
                self.unlabeled_mask = np.arange(
                    math.floor(len(self.true_Y) * self.TEST_FRACTION), len(self.true_Y)
                )
            self.test_mask = np.arange(
                0, math.floor(len(self.true_Y) * self.TEST_FRACTION)
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
                    random_index = np.where(self.true_Y[self.unlabeled_mask] == label)[
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

    def set_weak_supervisions(
        self,
        weak_supervisions: List[BaseWeakSupervision],
        merge_weak_supervision_label_strategy: BaseMergeWeakSupervisionLabelStrategy,
    ) -> None:
        self.merge_weak_supervision_label_strategy = (
            merge_weak_supervision_label_strategy
        )
        self.weak_supervisions = weak_supervisions

    def unlabel_samples(self, query_indices: IndiceMask) -> None:
        self.unlabeled_mask = np.append(self.unlabeled_mask, query_indices, axis=0)

        for sample in query_indices:
            self.labeled_mask = self.labeled_mask[self.labeled_mask != sample]

        self.human_expert_Y[query_indices] = -1
        self.Y_merged_final[query_indices] = -1

    def update_samples(self, query_indices: IndiceMask, Y_query: LabelList) -> None:
        self.human_expert_Y[query_indices] = Y_query
        self.Y_merged_final[query_indices] = Y_query

    def label_samples(
        self, query_indices: IndiceMask, Y_queries: LabelList, label_source: str
    ) -> None:
        self.labeled_mask = np.append(self.labeled_mask, query_indices, axis=0)  # type: ignore

        for element in query_indices:
            self.unlabeled_mask = self.unlabeled_mask[self.unlabeled_mask != element]

        self.label_source[query_indices] = label_source
        self.human_expert_Y[query_indices] = Y_queries
        self.Y_merged_final[query_indices] = Y_queries

        # @TODO later: eventuell basierend auf Daten mit dazu rechnen, dass Label verschiedene Kosten haben (Annotationsdauer in Experimenten etc)
        self.costs_spend += len(query_indices)

    def get_experiment_labels(self, query_indice: IndiceMask) -> LabelList:
        return self.true_Y[query_indice]

    def generate_weak_labels(self, learner: Learner, mask="unlabeled_mask") -> None:
        if len(self.weak_supervisions) == 0:
            print("No weak supervision strategies provided")
            exit(-1)

        if mask == "unlabeled_mask":
            mask = self.unlabeled_mask

        # store in weak_combined_Y
        # merge together with human_expert_Y into Y
        ws_labels_list: List[LabelList] = []
        for weak_supervision in self.weak_supervisions:
            ws_labels_list.append(weak_supervision.get_labels(mask, self, learner))

        # convert from [array([-1., -1., -1., ..., -1., -1., -1.]), array([-1., -1., -1., ..., -1., -1., -1.]), array([-1., -1., -1., ..., -1., -1., -1.]), array([-1., -1., -1., ..., -1., -1., -1.]), array([-1., -1., -1., ..., -1., -1., -1.])]

        # to [[-1,-1,4,2], [-1,4,-1], …]
        ws_labels_array: np.ndarray = np.transpose(np.array(ws_labels_list))

        self.ws_labels_list = ws_labels_array

        # magic
        self.weak_combined_Y[mask] = self.merge_weak_supervision_label_strategy.merge(
            ws_labels_array
        )

        # print(self.weak_combined_Y)

        # extract from self.weak_combined_Y only those who have not -1 and add them to the mask
        self.weakly_combined_mask = np.array(
            [indice for indice in mask if self.weak_combined_Y[indice] != -1]
        )

        self.only_weak_mask = np.array(
            [
                indice
                for indice in mask
                if self.weak_combined_Y[indice] != -1
                if indice not in self.labeled_mask
            ]
        )
        # merge labeled mask back into it
        self.weakly_combined_mask = np.array(
            list(set().union(self.labeled_mask, self.weakly_combined_mask))
        )

        # problem: when we directly use weak_combined_Y as labels -> we potentially a lot of -1!!!
        # first write WS labels
        self.Y_merged_final[mask] = self.weak_combined_Y[mask]

        # but later overwrite it with the true_Y labels
        self.Y_merged_final[self.labeled_mask] = self.true_Y[self.labeled_mask]

        # if all WS return -1 the following does not hold true
        # assert -1 not in self.Y_merged_final[self.only_weak_mask]
