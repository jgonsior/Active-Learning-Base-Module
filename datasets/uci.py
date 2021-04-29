from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_uci(
    DATASETS_PATH: str, DATASET_NAME: str, RANDOM_SEED: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(DATASETS_PATH + "/uci_cleaned/" + DATASET_NAME + ".csv")

    synthetic_creation_args = {}
    synthetic_creation_args["n_classes"] = len(df["LABEL"].unique())

    Y = df["LABEL"].to_numpy()
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    df["LABEL"] = Y
    df.rename(columns={"LABEL": "label"}, inplace=True)

    return df, synthetic_creation_args