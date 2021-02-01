import pandas as pd
import numpy as np


def load_alc(DATASETS_PATH: str, DATASET_NAME: str, RANDOM_SEED: int) -> pd.DataFrame:
    df = pd.read_csv(
        DATASETS_PATH + "/al_challenge/" + DATASET_NAME + ".data",
        header=None,
        sep=" ",
    )
    # feature_columns fehlt

    # shuffle df
    df = df.sample(frac=1, random_state=RANDOM_SEED)

    df = df.replace(str(-np.inf), "-1")
    df = df.replace(str(np.inf), "-1")
    df = df.fillna(0)

    labels = pd.read_csv(
        DATASETS_PATH + "/al_challenge/" + DATASET_NAME + ".label",
        header=None,
    )

    labels = labels.replace("-1", "A")
    labels = labels.replace("1", "B")
    df["label"] = labels

    return df
