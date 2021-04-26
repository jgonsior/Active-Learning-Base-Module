import numpy as np
import os
import pandas as pd
from scikeras.wrappers import KerasRegressor, KerasClassifier
from tensorflow import keras

from .BatchStateEncoding import BatchStateSampling
from .ImitationLearningBaseQuerySampler import (
    ImitationLearningBaseQuerySampler,
    InputState,
    OutputState,
    PreSampledIndices,
)
from .SingleStateEncoding import SingleStateEncoding
import joblib


class TrainedImitALSampler(ImitationLearningBaseQuerySampler):
    def __init__(self, NN_BINARY_PATH: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # reload the correct dimensions
        X = pd.read_csv(
            os.path.dirname(NN_BINARY_PATH) + "/01_state_encodings_X.csv", nrows=10
        )
        Y = pd.read_csv(
            os.path.dirname(NN_BINARY_PATH) + "/01_expert_actions_Y.csv", nrows=10
        )

        keras_model = keras.models.load_model(NN_BINARY_PATH)
        if WRAPPER == "regression":
            model_wrapper = KerasRegressor
        elif WRAPPER == "classifier":
            model_wrapper = KerasClassifier
        else:
            print("Only regression and classifier are allowd wrappers, exiting…")
            exit(-1)

        model = model_wrapper(keras_model)  # type: ignore
        model.initialize(X, Y)

        self.scaler = joblib.load(NN_BINARY_PATH + "_scaler.gz")

        # with open(NN_BINARY_PATH, "rb") as handle:
        #    model = pickle.load(handle)

        self.sampling_classifier = model

    def applyNN(self, X_input_state: InputState) -> OutputState:
        X_state = np.reshape(X_input_state, (1, len(X_input_state)))
        X_state = self.scaler.transform(X_state)
        Y_pred: OutputState = self.sampling_classifier.predict(X_state)
        return Y_pred

    def calculateImitationLearningData(
        self, pre_sampled_X_querie_indices: PreSampledIndices
    ) -> None:
        pass


class TrainedImitALSingleSampler(TrainedImitALSampler, SingleStateEncoding):
    pass


class TrainedImitALBatchSampler(TrainedImitALSampler, BatchStateSampling):
    pass
