from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib.dates as mdates
from storms import SolarWind
import argparse
from pathlib import Path
import numpy as np
import astropy.units as u

parser = argparse.ArgumentParser(description="Plot the txings GOES proxy.")
parser.add_argument("--out_path", type="str", description="The location to write the image.")
parser.add_argument("--days", type=float, default=3.0, description="The number of days to show on the plot.")
args = parser.parse_args()

out_path = Path(args.out_path if args.out_path else "/proj/web-cxc/htdocs/acis/")

plt.rc("font", size=18)
plt.rc("axes", linewidth=2)
plt.rc("xtick.major", size=6, width=2)
plt.rc("ytick.major", size=6, width=2)
plt.rc("xtick.minor", size=3, width=2)
plt.rc("ytick.minor", size=3, width=2)


p = Path("/data/acis/txings/txings_proxy.fits")
t_proxy = Table.read(p)

times = Time(t_proxy["time"], format='cxcsec')

sw = SolarWind(times[0].yday, times[-1].yday, get_txings=True)

print(times[-1].yday)

idxs = times >= times[-1] - 3.0*u.day

fi_rate_limit = 320.0*np.ones_like(times.cxcsec)

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(times.datetime[idxs], t_proxy["fi_rate_predict"][idxs], 'x', label="Prediction", color="C0")
ax.plot(times.datetime[idxs], fi_rate_limit[idxs], '.', label="Limit", color="C0")
ax.plot(sw.txings_times.datetime, sw.txings_data["fi_rate"], '.', label="Data", color="C1")
ax.set_xlabel("Date")
ax.legend()
ax.set_ylabel("ACIS Threshold Crossing Rate (cts/sec/row)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%j:%H:%M:%S'))
fig.autofmt_xdate()
ax.grid()
ax.set_xlim(times.datetime[idxs][0], None)
ax.set_ylim(None, 400.0)
fig.savefig(out_path / "fi_rate_predict.png", bbox_inches='tight')
