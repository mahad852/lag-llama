import numpy as np


class ECG_MIT:
    def __init__(self, 
                 context_len = 64, 
                 pred_len = 64, 
                 data_path = "MIT-BIH.npz"):
        self.context_len = context_len
        self.pred_len = pred_len
        self.data_path = data_path

        self.data = np.zeros((650000, 46))

        self.total_len = (self.data.shape[0] - self.pred_len - self.context_len + 1) * self.data.shape[1]

        self.load_dataset()

    def load_dataset(self):
        with np.load(self.data_path) as d:
            for i, file in enumerate(d.files):
                self.data[:, i] = d[file]

    def __getitem__(self, index):
        ind2 = index // (self.data.shape[0] - self.pred_len - self.context_len + 1)
        ind1 = index % (self.data.shape[0] - self.pred_len - self.context_len + 1)

        x = self.data[ind1:ind1 + self.context_len, ind2]
        y = self.data[ind1 + self.context_len: ind1 + self.context_len + self.pred_len, ind2]

        return x, y

    def __len__(self):
        return self.total_len