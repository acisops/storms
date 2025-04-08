from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib.dates as mdates
from storms import SolarWind
from cxotime import CxoTime
from kadi.events import obsids, rad_zones
import argparse
from pathlib import Path
import numpy as np
import astropy.units as u

parser = argparse.ArgumentParser(description="Plot the txings GOES proxy.")
parser.add_argument("--out_path", type=str, help="The location to write the image.")
parser.add_argument("--days", type=float, default=3.0, help="The number of days to show on the plot.")
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

sw = SolarWind(times[0].yday, times[-1].yday, get_txings=True, get_ace=False)

print(times[-1].yday)

idxs = times >= times[-1] - 3.0*u.day

fi_rate_limit = 320.0*np.ones_like(times.cxcsec)

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(times.datetime[idxs], t_proxy["fi_rate_predict"][idxs], 'x', ms=8, mew=3, label="FI Prediction", color="C0")
ax.plot(times.datetime[idxs], fi_rate_limit[idxs], label="FI Limit", lw=2, color="C0")
ax.plot(sw.txings_times.datetime, sw.txings_data["fi_rate"], '.', ms=6, label="FI Data", color="C1")
ax.set_xlabel("Date")
ax.legend()
ax.set_ylabel("ACIS Threshold Crossing Rate (cts/sec/100 rows)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%j:%H:%M:%S'))
fig.autofmt_xdate()
ax.grid()
ax.set_xlim(times.datetime[idxs][0], (times[-1]+0.5*u.day).datetime)
ax.set_ylim(None, 400.0)
for o in obsids.filter(start=times.yday[idxs][0], stop=(times[-1]+0.5*u.day).yday):
    x = CxoTime(0.5*(o.tstart+o.tstop))
    if x >= times.datetime[idxs][0]:
        ax.text(x.datetime, 370, str(o.obsid), color="C3", rotation=90, ha='center', va='bottom', fontsize=14)
for rz in rad_zones.filter(start=times.yday[idxs][0], stop=(times[-1]+2*u.day).yday):
    ax.axvspan(CxoTime(rz.tstart).datetime, 
               CxoTime(rz.tstop).datetime,
               color="mediumpurple", alpha=0.333333)
fig.savefig(out_path / "fi_rate_predict.png", bbox_inches='tight')
