import joblib
import numpy as np
import torch
from argparse import ArgumentParser
from astropy.table import Table, vstack
from cheta import fetch_sci as fetch
from cxotime import CxoTime
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from storms.txings_proxy.realtime import get_realtime_goes
from storms.txings_proxy.utils import goes_bands, MLPModel
from storms.utils import base_path

models_path = base_path / "txings_proxy/Models"

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

input_length = 32

scaler_fi_x = joblib.load(models_path / "scaler_fi_rate_x.pkl")
scaler_fi_y = joblib.load(models_path / "scaler_fi_rate_y.pkl")

scaler_bi_x = joblib.load(models_path / "scaler_bi_rate_x.pkl")
scaler_bi_y = joblib.load(models_path / "scaler_bi_rate_y.pkl")

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
        break_time = t_exist["time"][-1]
        use7d = args.use_7day
    else:
        print("overwrite")
        t_exist = None
        break_time = -1.0
        use7d = True
    t_goes, first_time = get_realtime_goes(break_time, use7d=use7d)
    if t_exist is not None:
        if t_goes is None or t_exist["time"][-1] < first_time:
            t_goes, first_time = get_realtime_goes(break_time, use7d=True)
        if t_goes is None:
            t_goes = t_exist
        else:
            # Remove values that are less than zero
            good = np.ones(len(t_goes), dtype=bool)
            for col in t_goes.columns:
                if col.startswith("P"):
                    good &= t_goes[col] >= 0.0
            if np.sum(good) == 0:
                t_goes = t_exist
            else:
                for col in t_goes.columns:
                    if col.startswith("P"):
                        t_goes[col][~good] = np.nan
                t_goes = vstack([t_exist, t_goes])

ephem_msids = ['orbitephem0_x', 'orbitephem0_y', 'orbitephem0_z', 'solarephem0_x', 'solarephem0_y', 'solarephem0_z']
ephem_data = fetch.MSIDset(ephem_msids, t_goes["time"][0], t_goes["time"][-1], stat="5min")

print(len(t_goes["time"]))

for msid in ephem_msids:
    t_goes[msid] = np.interp(t_goes["time"], ephem_data[msid].times, ephem_data[msid].vals)

use_cols = ['P1_g16_E', 'P2A_g16_E', 'P2B_g16_E', 'P3_g16_E', 'P4_g16_E',
            'P5_g16_E', 'P6_g16_E', 'P7_g16_E', 'P8A_g16_E', 'P8B_g16_E',
            'P8C_g16_E', 'P9_g16_E', 'P10_g16_E', 'P1_g18_E', 'P2A_g18_E',
            'P2B_g18_E', 'P3_g18_E', 'P4_g18_E', 'P5_g18_E', 'P6_g18_E', 'P7_g18_E',
            'P8A_g18_E', 'P8B_g18_E', 'P8C_g18_E', 'P9_g18_E', 'P10_g18_E',
            'orbitephem0_x', 'orbitephem0_y', 'orbitephem0_z', 'solarephem0_x', 'solarephem0_y', 'solarephem0_z']

df = t_goes[use_cols].to_pandas()

# Convert the specific flux to total flux in the band for each channel by multiplying
# by the channel width
for col in df.columns:
    if col.startswith("P"):
        prefix = col.split("_")[0]
        i = use_cols.index(col)
        df[col] = uniform_filter1d(df[col], 10, axis=0) * (goes_bands[prefix][1] - goes_bands[prefix][0])

X = np.array(df)

n_folds = 10

xx_fi = torch.from_numpy(scaler_fi_x.transform(X)).to(device, torch.float32)
xx_bi = torch.from_numpy(scaler_bi_x.transform(X)).to(device, torch.float32)

y_inv_fi = []
y_inv_bi = []
for k in range(n_folds):
    model_fi = MLPModel(input_length).to(device)
    model_fi.load_state_dict(torch.load(models_path / f"fi_rate_k{k}_model", map_location='cpu'))
    model_bi = MLPModel(input_length).to(device)
    model_bi.load_state_dict(torch.load(models_path / f"bi_rate_k{k}_model", map_location='cpu'))
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
