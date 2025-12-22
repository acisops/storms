from argparse import ArgumentParser
from pathlib import Path

import astropy.units as u
import joblib
from astropy.table import Table, vstack
from cxotime import CxoTime

from storms.txings_proxy.realtime import get_realtime_goes
from storms.txings_proxy.utils import prep_data, run_model


parser = ArgumentParser(
    description="Make the real-time GOES-based proxy for ACIS txings."
)

parser.add_argument(
    "--out_file",
    type=str,
    default="/data/acis/txings/txings_proxy.fits",
    help="The path of the file to be written.",
)

parser.add_argument("--use_7day", action="store_true", help="Use 7-day file.")

parser.add_argument(
    "--overwrite_table", action="store_true", help="Overwrite the table."
)

args = parser.parse_args()


p = Path(args.out_file)
start_over = not p.exists() or args.overwrite_table
use7d = start_over or args.use_7day
try:
    t_goes = get_realtime_goes(use7d=use7d)
except RuntimeError as e1:
    try:
        t_goes = get_realtime_goes(use7d=True)
    except RuntimeError as e2:
        raise e2 from e1
ephem_stop = CxoTime()
ephem_start = ephem_stop - 8.0 * u.day
if not start_over:
    t_exist = Table.read(p)
    t_exist = t_exist[t_exist["time"] < t_goes["time"][0]]
else:
    t_exist = None


X = prep_data(t_goes)

for which_rate in ["fi_rate", "bi_rate"]:
    t_goes[f"{which_rate}_predict"] = run_model(X, which_rate)

if t_exist is None:
    t_final = t_goes
else:
    t_final = vstack([t_exist, t_goes])
t_final.write(p, overwrite=True)
