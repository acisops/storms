from pathlib import Path

import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler


goes_path = Path("/data/acis/goes")

goes_bands = {
    "P1": [1.02, 1.86],
    "P2A": [1.9, 2.3],
    "P2B": [2.31, 3.34],
    "P3": [3.4, 6.48],
    "P4": [5.84, 11.0],
    "P5": [11.64, 23.27],
    "P6": [25.9, 38.1],
    "P7": [40.3, 73.4],
    "P8A": [83.7, 98.5],
    "P8B": [99.9, 118.0],
    "P8C": [115.0, 143.0],
    "P9": [160.0, 242.0],
    "P10": [276.0, 404.0],
}


def transform_goes(t_in):
    # Remove values that are less than or equal to zero
    good = np.ones(len(t_in), dtype=bool)
    for col in t_in.columns:
        if col.startswith("P"):
            good &= t_in[col] > 0.0
    t_out = t_in[good]

    # Convert the specific flux to total flux in the band for each channel by multiplying
    # by the channel width
    for col in t_out.columns:
        if col.startswith("P"):
            prefix = col.split("_")[0]
            t_out[col] *= (goes_bands[prefix][1] - goes_bands[prefix][0])

    return t_out


class LogHyperbolicTangentScaler:
    def __init__(self):
        self._mean = 1.0

    def fit(self, x):
        self._mean = np.mean(np.array(x), axis=0)

    def transform(self, x):
        return np.tanh(np.log10(x / self._mean + 1.0))

    def inverse_transform(self, y):
        return self._mean*(10**np.arctanh(y) - 1.0)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class MultiScaler:
    """Applies different sklearn scalers to different column subsets."""

    def __init__(self, colnames):
        is_proton = np.char.startswith(colnames, "P")
        self.lht_cols = np.where(is_proton)[0]
        self.mm_cols = np.where(~is_proton)[0]

        self.lht_scaler = LogHyperbolicTangentScaler()
        self.mm_scaler = MinMaxScaler()

    def fit(self, X):
        self.lht_scaler.fit(X[:, self.lht_cols])
        self.mm_scaler.fit(X[:, self.mm_cols])

    def transform(self, X):
        X = X.copy()
        X[:, self.lht_cols] = self.lht_scaler.transform(X[:, self.lht_cols])
        X[:, self.mm_cols] = self.mm_scaler.transform(X[:, self.mm_cols])
        return X

    def inverse_transform(self, y):
        y = y.copy()
        y[:, self.lht_cols] = self._mean*(10**np.arctanh(y[:, self.lht_cols]) - 1.0)
        y[:, self.mm_cols] = self.mm_scaler.inverse_transform(y[:, self.mm_cols])
        return y

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MLPModel(nn.Module):
    def __init__(self, input_length, x_factor=10):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_length, input_length*2*x_factor),
            nn.LeakyReLU(),
            nn.Linear(input_length*2*x_factor, input_length*4*x_factor),
            nn.LeakyReLU(),
            nn.Linear(input_length*4*x_factor, input_length*2*x_factor),
            nn.LeakyReLU(),
            nn.Linear(input_length*2*x_factor, input_length*x_factor),
            nn.LeakyReLU(),
            nn.Linear(input_length*x_factor, 1)
        )

    def forward(self, x):
        return self.model(x)
