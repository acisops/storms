import torch.nn as nn
import torch
from astropy.table import Table, vstack
from storms.utils import base_path
from storms.txings_proxy.utils import goes_bands, LogHyperbolicTangentScaler, transform_goes
from storms.realtime import get_realtime_goes

from pathlib import Path
import numpy as np

data_path = base_path / "txings_proxy/data"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

overwrite_table = False

x_factor = 10
INPUT_LENGTH = 26

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_LENGTH, INPUT_LENGTH*2*x_factor),
            nn.LeakyReLU(),
            nn.Linear(INPUT_LENGTH*2*x_factor, INPUT_LENGTH*4*x_factor),
            nn.LeakyReLU(),
            nn.Linear(INPUT_LENGTH*4*x_factor, INPUT_LENGTH*2*x_factor),
            nn.LeakyReLU(),
            nn.Linear(INPUT_LENGTH*2*x_factor, INPUT_LENGTH*x_factor),
            nn.LeakyReLU(),
            nn.Linear(INPUT_LENGTH*x_factor, 1)
        )

    def forward(self, x):
        return self.model(x)


means = np.load(data_path / "means.npz")

scaler_x = LogHyperbolicTangentScaler(mean=means["x"])
scaler_y = LogHyperbolicTangentScaler(mean=means["y"])

p = Path("/data/acis/txings/txings_proxy.fits")
if p.exists() and not overwrite_table:
    t_exist = Table.read(p)
    start_time = t_exist["time"][-1]
    use7d = True
else:
    print("overwrite")
    t_exist = None
    start_time = -1.0
    use7d = True
t_goes = get_realtime_goes(start_time, use7d=use7d)
if t_exist is not None:
    if t_goes is None:
        t_goes = t_exist
    else:
        t_goes = vstack([t_exist, t_goes])

# Remove values that are less than or equal to zero
good = np.ones(len(t_goes), dtype=bool)
for col in t_goes.columns:
    if col.startswith("P"):
        good &= t_goes[col] > 0.0
t_goes = t_goes[good]

#use_chans = list(goes_bands.keys())
#use_cols = [f"{chan}_g{source}_E" for chan in use_chans for source in [16, 18]]
use_cols = ['P1_g16_E', 'P2A_g16_E', 'P2B_g16_E', 'P3_g16_E', 'P4_g16_E',
       'P5_g16_E', 'P6_g16_E', 'P7_g16_E', 'P8A_g16_E', 'P8B_g16_E',
       'P8C_g16_E', 'P9_g16_E', 'P10_g16_E', 'P1_g18_E', 'P2A_g18_E',
       'P2B_g18_E', 'P3_g18_E', 'P4_g18_E', 'P5_g18_E', 'P6_g18_E', 'P7_g18_E',
       'P8A_g18_E', 'P8B_g18_E', 'P8C_g18_E', 'P9_g18_E', 'P10_g18_E']
df = t_goes[use_cols].to_pandas()

# Convert the specifix flux to total flux in the band for each channel by multiplying
# by the channel width
for col in t_goes.columns:
    if col.startswith("P"):
        prefix = col.split("_")[0]
        df[col] *= (goes_bands[prefix][1] - goes_bands[prefix][0])

X = np.array(df)
X = scaler_x.transform(X)
 
n_folds = 10

y_inv = []
for k in range(n_folds):
    MODEL_NAME = f"input_type_All_P_direction_East_fi_rate_k{k}_10x_model_1000_learning_rate_max_stop_50_DATA_AUG_1x"
    model = MLPModel().to(device)
    model.load_state_dict(torch.load(data_path / f"{MODEL_NAME}_min_epoch"))
    with torch.no_grad():
        model.eval()
        xx = torch.from_numpy(X).to(device, torch.float32)
        yy = model(xx).squeeze().cpu().detach().numpy()
    y_inv.append(scaler_y.inverse_transform(yy))

t_goes["fi_rate_predict"] = np.mean(y_inv, axis=0)
t_goes.write(p, overwrite=True)
