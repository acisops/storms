from pathlib import Path

import numpy as np
from torch import nn

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

coeffs_16_19 = {
    "P1": 0.8523294787577307,
    "P2A": 0.6829297841361497,
    "P2B": 0.9155916354664245,
    "P3": 0.7235568865422375,
    "P4": 1.5512339373799464,
    "P5": 1.171415803057102,
    "P6": 0.6621313006217242,
    "P7": 0.9829375532604085,
    "P8A": 0.9876683223951546,
    "P8B": 1.0428979147767126,
    "P8C": 0.5502883072599938,
    "P9": 0.756540925797541,
    "P10": 0.7197311497043392,
}


def transform_goes(t_in):
    # Remove values that are less than or equal to zero
    good = np.ones(len(t_in), dtype=bool)
    for col in t_in.columns:
        if col.startswith("P"):
            good &= t_in[col] > 0.0
    t_out = t_in[good]

    # Convert the specifix flux to total flux in the band for each channel by multiplying
    # by the channel width
    for col in t_out.columns:
        if col.startswith("P"):
            prefix = col.split("_")[0]
            t_out[col] *= (goes_bands[prefix][1] - goes_bands[prefix][0])

    return t_out


class LogHyperbolicTangentScaler:
    def __init__(self, mean=1.0):
        self._mean = mean

    def fit(self, x):
        self._mean = np.mean(np.array(x), axis=0)

    def transform(self, x):
        return np.tanh(np.log10(x / self._mean + 1.0))

    def inverse_transform(self, x):
        return self._mean*(10**np.arctanh(x) - 1.0)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class MLPModel(nn.Module):
    def __init__(self, input_length, x_factor):
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
