import torch.nn as nn
import torch
from astropy.table import Table, vstack
from storms.utils import base_path
from storms.txings_proxy.utils import goes_bands, LogHyperbolicTangentScaler, transform_goes
from storms.realtime import get_realtime_goes
from string import Template
from scipy.ndimage import uniform_filter1d

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


means_fi = np.load(data_path / "means_fi_rate.npz")
scaler_fi_x = LogHyperbolicTangentScaler(mean=means_fi["x"])
scaler_fi_y = LogHyperbolicTangentScaler(mean=means_fi["y"])

means_bi = np.load(data_path / "means_bi_rate.npz")
scaler_bi_x = LogHyperbolicTangentScaler(mean=means_bi["x"])
scaler_bi_y = LogHyperbolicTangentScaler(mean=means_bi["y"])

p = Path("/data/acis/txings/txings_proxy.fits")
if p.exists() and not overwrite_table:
    t_exist = Table.read(p)
    start_time = t_exist["time"][-1]
    use7d = False
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
"""
p = Path("/data/acis/txings/txings_proxy2.fits")
t = Table.read("/data/acis/goes/goes_16_18.fits")
print(t.colnames)
t_goes = Table()
for col in t.colnames:
    if col.startswith("P"):
        t_goes[f"{col}_E"] = t[col][:, 0]
t_goes["time"] = t["time"]  
"""

# Remove values that are less than zero
good = np.ones(len(t_goes), dtype=bool)
for col in t_goes.columns:
    if col.startswith("P"):
        good &= t_goes[col] >= 0.0
t_goes = t_goes[good]
for col in t_goes.columns:
    if col.startswith("P"):
        t_goes[col] = uniform_filter1d(t_goes[col], 10, axis=0, mode="nearest")

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
for col in df.columns:
    if col.startswith("P"):
        prefix = col.split("_")[0]
        df[col] *= (goes_bands[prefix][1] - goes_bands[prefix][0])

X = np.array(df)
 
n_folds = 10

xx_fi = torch.from_numpy(scaler_fi_x.transform(X)).to(device, torch.float32)
xx_bi = torch.from_numpy(scaler_bi_x.transform(X)).to(device, torch.float32)

model_template = Template("${which}_rate_k${fold}_10x_model_1000_learning_rate_max_stop_50_DATA_AUG_1x_min_epoch")
y_inv_fi = []
y_inv_bi = []
for k in range(n_folds):
    model_fi = MLPModel().to(device)
    model_fi.load_state_dict(torch.load(data_path / model_template.substitute(which="fi", fold=k), map_location='cpu'))
    model_bi = MLPModel().to(device)
    model_bi.load_state_dict(torch.load(data_path / model_template.substitute(which="bi", fold=k), map_location='cpu'))
    with torch.no_grad():
        model_fi.eval()
        model_bi.eval()
        yy_fi = model_fi(xx_fi).squeeze().cpu().detach().numpy()
        yy_bi = model_bi(xx_bi).squeeze().cpu().detach().numpy()
    y_inv_fi.append(scaler_fi_y.inverse_transform(yy_fi))
    y_inv_bi.append(scaler_bi_y.inverse_transform(yy_bi))

t_goes["fi_rate_predict"] = np.mean(y_inv_fi, axis=0)
t_goes["bi_rate_predict"] = np.mean(y_inv_bi, axis=0)
t_goes.write(p, overwrite=True)
