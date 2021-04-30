from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_uci(
    DATASETS_PATH: str, DATASET_NAME: str, RANDOM_SEED: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(DATASETS_PATH + "/uci_cleaned/" + DATASET_NAME + ".csv")

    synthetic_creation_args = synthetic_creation_args = {
        "n_samples": len(df),
        "n_features": len(df.columns),  # type: ignore
        "n_informative": len(df.columns),  # type: ignore
        "n_redundant": 0,  # type: ignore
        "n_repeated": 0,  # type: ignore
        "n_classes": len(df["LABEL"].unique()),  # type: ignore
        "n_clusters_per_class": "?",  # type: ignore
        "weights": "?",  # type: ignore
        "flip_y": "?",  # type: ignore
        "class_sep": "?",  # type: ignore
        "hypercube": "?",
        "scale": "?",  # type: ignore
        "random_state": RANDOM_SEED,
    }

    synthetic_creation_args["n_classes"] = len(df["LABEL"].unique())

    Y = df["LABEL"].to_numpy()
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    df["LABEL"] = Y
    df.rename(columns={"LABEL": "label"}, inplace=True)

    return df, synthetic_creation_args