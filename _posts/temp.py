class MNISTDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, transform=None, is_test=False):
        super(MNISTDataset, self).__init__()
        dataset = []
        labels_positive = {}
        labels_negative = {}
        if is_test == False:
            for i in list(data_df.label.unique()):
                labels_positive[i] = data_df[data_df.label == i].to_numpy()
            for i in list(data_df.label.unique()):
                labels_negative[i] = data_df[data_df.label != i].to_numpy()

        for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
            data = row.to_numpy()
            if is_test:
                label = -1
                first = data.reshape(28, 28)
                second = -1
                dis = -1
            else:
                label = data[0]
                first = data[1:].reshape(28, 28)
                if np.random.randint(0, 2) == 0:
                    second = labels_positive[label][
                        np.random.randint(0, len(labels_positive[label]))
                    ]
                else:
                    second = labels_negative[label][
                        np.random.randint(0, len(labels_negative[label]))
                    ]
                dis = 1.0 if second[0] == label else 0.0
                second = second[1:].reshape(28, 28)

            if transform is not None:
                first = transform(first.astype(np.float32))
                if second is not -1:
                    second = transform(second.astype(np.float32))

            dataset.append((first, second, dis, label))
        self.dataset = dataset
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
