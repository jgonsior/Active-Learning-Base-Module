import json
import os
import dill
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


class ALiPY_ImitAL_Query_Strategy:
    trained_imitAL_sampler: TrainedImitALSampler
    X: np.ndarray
    Y: np.ndarray

    def __init__(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        self.X = X
        self.Y = Y

        if "OLD_PATH" in kwargs.keys():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

            with open(kwargs["NN_BINARY_PATH"], "rb") as handle:
                model = dill.load(handle)

            self.sampling_classifier = model

        # load NN params from json file
        with open(
            os.path.dirname(kwargs["NN_BINARY_PATH"])
            + "/01_dataset_creation_stats.csv_params.json",
            "r",
        ) as f:
            content = str(f.read())
            dataset_stats = json.loads(content)

        # print(dataset_stats)

        # overwrite
        params = {
            "PRE_SAMPLING_METHOD": dataset_stats["PRE_SAMPLING_METHOD"],
            "PRE_SAMPLING_ARG": dataset_stats["PRE_SAMPLING_ARG"],
            "AMOUNT_OF_PEAKED_OBJECTS": dataset_stats["AMOUNT_OF_PEAKED_OBJECTS"],
            "DISTANCE_METRIC": dataset_stats["DISTANCE_METRIC"],
            "STATE_ARGSECOND_PROBAS": dataset_stats["STATE_ARGSECOND_PROBAS"],
            "STATE_ARGTHIRD_PROBAS": dataset_stats["STATE_ARGTHIRD_PROBAS"],
            "STATE_DIFF_PROBAS": dataset_stats["STATE_DIFF_PROBAS"],
            "STATE_PREDICTED_CLASS": dataset_stats["STATE_PREDICTED_CLASS"],
            "STATE_DISTANCES_LAB": dataset_stats["STATE_DISTANCES_LAB"],
            "STATE_DISTANCES_UNLAB": dataset_stats["STATE_DISTANCES_UNLAB"],
            "STATE_INCLUDE_NR_FEATURES": dataset_stats["STATE_INCLUDE_NR_FEATURES"],
            "NN_BINARY_PATH": kwargs["NN_BINARY_PATH"],
        }
        data_storage = kwargs["data_storage"]
        del kwargs["data_storage"]
        params.update(kwargs)

        self.trained_imitAL_sampler = TrainedImitALSingleSampler(**params)

        self.trained_imitAL_sampler.data_storage = data_storage

    def select(self, labeled_index, unlabeled_index, model, batch_size=1, **kwargs):
        self.trained_imitAL_sampler.learner = model
        self.trained_imitAL_sampler.data_storage.labeled_mask = labeled_index
        self.trained_imitAL_sampler.data_storage.unlabeled_mask = unlabeled_index

        # update data_storage with labeled_index and unlabeled_index
        pre_sampled_X_querie_indices: PreSampledIndices = (
            self.trained_imitAL_sampler.pre_sample_potential_X_queries()
        )

        # when using a pre-trained model this does nothing
        self.trained_imitAL_sampler.calculateImitationLearningData(
            pre_sampled_X_querie_indices
        )

        X_input_state: InputState = self.trained_imitAL_sampler.encode_input_state(
            pre_sampled_X_querie_indices
        )
        Y_output_state: OutputState = self.trained_imitAL_sampler.applyNN(
            X_input_state
        )[0]
        """ print()
        print("labeled are ", sorted(labeled_index))
        print(sorted(pre_sampled_X_querie_indices))
        print(
            "selected: ",
            sorted(
                [
                    v
                    for v in self.trained_imitAL_sampler.decode_output_state(
                        Y_output_state, pre_sampled_X_querie_indices, batch_size
                    )
                ]
            ),
        )
        print() """

        return [
            v
            for v in self.trained_imitAL_sampler.decode_output_state(
                Y_output_state, pre_sampled_X_querie_indices, batch_size
            )
        ]