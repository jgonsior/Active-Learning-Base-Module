
 def _load_alc(self):
        df = pd.read_csv(
            DATASETS_PATH + "/al_challenge/" + self.DATASET_NAME + ".data",
            header=None,
            sep=" ",
        )
        # feature_columns fehlt

        # shuffle df
        df = df.sample(frac=1, random_state=self.RANDOM_SEED)

        df = df.replace([np.inf, -np.inf], -1)
        df = df.fillna(0)

        labels = pd.read_csv(
            self.DATASETS_PATH + "/al_challenge/" + self.DATASET_NAME + ".label",
            header=None,
        )

        labels = labels.replace([-1], "A")
        labels = labels.replace([1], "B")

        return df.to_numpy(), labels[0].to_numpy()

