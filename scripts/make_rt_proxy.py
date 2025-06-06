import torch.nn as nn
import torch
from astropy.table import Table, vstack
from storms.utils import base_path
from storms.txings_proxy.utils import goes_bands, LogHyperbolicTangentScaler, transform_goes
from storms.realtime import get_realtime_goes
from string import Template
from scipy.ndimage import uniform_filter1d
from argparse import ArgumentParser
from cxotime import CxoTime

from pathlib import Path
import numpy as np

data_path = base_path / "txings_proxy/data"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

parser = ArgumentParser(
    description="Make the real-time GOES-based proxy for ACIS txings."
)

parser.add_argument(
    "--out_file", type=str, default="/data/acis/txings/txings_proxy.fits",
    help="The path of the file to be written."
)

parser.add_argument(
    "--use_historical", action="store_true", 
    help="Use historical data."
)

parser.add_argument(
    "--use_7day", action="store_true", 
    help="Use 7-day file."
)

parser.add_argument(
    "--overwrite_table", action="store_true",
    help="Overwrite the table."
)

parser.add_argument(
    "--start", type=str,
    help="The start time, if using historical data."
)

parser.add_argument(
    "--stop", type=str,
    help="The stop time, if using historical data."
)

args = parser.parse_args()

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


p = Path(args.out_file)
if args.use_historical:
    t = Table.read("/data/acis/goes/goes_16_18.fits")
    if args.start is not None:
        tstart = CxoTime(args.start).secs
    else:
        tstart = t["time"][0] - 1.0
    if args.start is not None:
        tstop = CxoTime(args.stop).secs
    else:
        tstop = t["time"][-1] + 1.0
    idxs = (t["time"] >= tstart) & (t["time"] <= tstop)
    t_goes = Table()
    for col in t.colnames:
        if col.startswith("P"):
            t_goes[f"{col}_E"] = t[col][idxs, 0]
    t_goes["time"] = t["time"][idxs]
    # Remove values that are less than zero
    good = np.ones(len(t_goes), dtype=bool)
    for col in t_goes.columns:
        if col.startswith("P"):
            good &= t_goes[col] >= 0.0
    t_goes = t_goes[good]
else:
    if p.exists() and not args.overwrite_table:
        t_exist = Table.read(p)
        start_time = t_exist["time"][-1]
        use7d = args.use_7day
    else:
        print("overwrite")
        t_exist = None
        start_time = -1.0
        use7d = True
    t_goes = get_realtime_goes(start_time, use7d=use7d)
    # Remove values that are less than zero
    good = np.ones(len(t_goes), dtype=bool)
    for col in t_goes.columns:
        if col.startswith("P"):
            good &= t_goes[col] >= 0.0
    t_goes = t_goes[good]
    if t_exist is not None:
        if t_goes is None:
            t_goes = t_exist
        else:
            t_goes = vstack([t_exist, t_goes])

use_cols = ['P1_g16_E', 'P2A_g16_E', 'P2B_g16_E', 'P3_g16_E', 'P4_g16_E',
       'P5_g16_E', 'P6_g16_E', 'P7_g16_E', 'P8A_g16_E', 'P8B_g16_E',
       'P8C_g16_E', 'P9_g16_E', 'P10_g16_E', 'P1_g18_E', 'P2A_g18_E',
       'P2B_g18_E', 'P3_g18_E', 'P4_g18_E', 'P5_g18_E', 'P6_g18_E', 'P7_g18_E',
       'P8A_g18_E', 'P8B_g18_E', 'P8C_g18_E', 'P9_g18_E', 'P10_g18_E']
df = t_goes[use_cols].to_pandas()

weights = np.ones(len(use_cols))
weights[use_cols.index('P4_g16_E')] = 2.0
weights[use_cols.index('P6_g16_E')] = 0.7
#weights[use_cols.index('P8C_g16_E')] = 0.5

# Convert the specifix flux to total flux in the band for each channel by multiplying
# by the channel width
for col in df.columns:
    if col.startswith("P"):
        prefix = col.split("_")[0]
        i = use_cols.index(col)
        df[col] = weights[i]*uniform_filter1d(df[col], 10, axis=0)*(goes_bands[prefix][1] - goes_bands[prefix][0])
        
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
