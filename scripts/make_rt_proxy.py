from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from astropy.table import Table
from cxotime import CxoTime

from storms.txings_proxy.utils import prep_data, run_model

parser = ArgumentParser(
    description="Make the GOES-based proxy for ACIS txings from historical data."
)

parser.add_argument("start", type=str, help="The start time.")

parser.add_argument("stop", type=str, help="The stop time.")

parser.add_argument(
    "--out_file",
    type=str,
    default="/data/acis/txings/txings_proxy.fits",
    help="The path of the file to be written.",
)

parser.add_argument(
    "--overwrite_table", action="store_true", help="Overwrite the table."
)

args = parser.parse_args()

p = Path(args.out_file)

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
ephem_start = tstart
ephem_stop = tstop


X = prep_data(t_goes)

for which_rate in ["fi_rate", "bi_rate"]:
    t_goes[f"{which_rate}_predict"] = np.mean(run_model(X, which_rate), axis=0)

t_goes.write(p, overwrite=True)
