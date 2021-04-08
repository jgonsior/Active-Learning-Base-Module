import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dwtc(DATASETS_PATH: str, RANDOM_SEED: int) -> pd.DataFrame:

    df = pd.read_csv(DATASETS_PATH + "dwtc/aft.csv")

    # shuffle df
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    #  self.synthetic_creation_args = {}
    #  self.synthetic_creation_args["n_classes"] = len(df["CLASS"].unique())

    Y = df["CLASS"].to_numpy()
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    df["label"] = Y
    df = df.drop("CLASS", 1)

    return df
