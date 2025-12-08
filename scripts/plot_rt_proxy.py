from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib.dates as mdates
from storms import SolarWind
from cxotime import CxoTime
from kadi.events import obsids, rad_zones
from kadi.commands.states import get_states
import argparse
from pathlib import Path
import numpy as np
import astropy.units as u

parser = argparse.ArgumentParser(description="Plot the txings GOES proxy.")

parser.add_argument("--infile", type=str, default="/data/acis/txings/txings_proxy.fits", help="The file containing the proxy to read.")
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


p = Path(args.infile)

t_proxy = Table.read(p)

times = Time(t_proxy["time"], format='cxcsec')

tstop = times[-1]
tstart = times[-1] - args.days*u.day

print(tstart.yday, tstop.yday)

sw = SolarWind(tstart.yday, tstop.yday, get_txings=True, get_ace=False)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 13), constrained_layout=True)

model_idxs = times >= tstop - args.days*u.day
model_times = times.datetime[model_idxs]
fi_rate_predict = t_proxy["fi_rate_predict"][model_idxs]
bi_rate_predict = t_proxy["bi_rate_predict"][model_idxs]
fi_rate_limit = 320.0*np.ones_like(times.cxcsec)[model_idxs]
bi_rate_limit = 20.0*np.ones_like(times.cxcsec)[model_idxs]

data_idxs = sw.txings_times >= tstop - args.days*u.day
data_times = sw.txings_times.datetime[data_idxs]
fi_rate = sw.txings_data["fi_rate"][data_idxs]
bi_rate = sw.txings_data["bi_rate"][data_idxs]

ax1.plot(model_times, fi_rate_predict, lw=3, label="FI Prediction", color="C0")
ax1.plot(model_times, fi_rate_limit, label="FI Limit", lw=2, color="C0")
ax1.plot(data_times, fi_rate, '.', ms=6, label="FI Data", color="C1")

ax2.plot(model_times, bi_rate_predict, lw=3, label="BI Prediction", color="C2")
ax2.plot(model_times, bi_rate_limit, label="BI Limit", lw=2, color="C2")
ax2.plot(data_times, bi_rate, '.', ms=6, label="BI Data", color="C3")

for ax in [ax1, ax2]:
    ax.grid()
    ax.set_xlim(model_times[0], times[-1].datetime)
    ax.set_xlabel("Date")
    ax.legend(fontsize=16, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%j:%H"))

fig.autofmt_xdate()

ax1.set_ylim(80, max(420.0, fi_rate.max()*1.1, fi_rate_predict.max()*1.1))
ax2.set_ylim(5, max(35.0, bi_rate.max()*1.1, bi_rate_predict.max()*1.1))

fig.supylabel("ACIS Threshold Crossing Rate (cts/sec/100 rows)")

states = get_states(start=times.yday[model_idxs][0], stop=times[-1].yday,
                    merge_identical=True)
oidxs = np.array(["obsid" in s for s in states["trans_keys"]])
obsids = states["obsid"][oidxs]
tstart = CxoTime(states["tstart"][oidxs])

for o, t in zip(obsids, tstart):
    if t >= model_times[0] and o <= 39000:
        for ax, ypos in zip([ax1, ax2], [100, 8]):
            ax.text(t.datetime, ypos, str(o), color="C3", rotation=90, 
                    ha='center', va='bottom', fontsize=14, zorder=-100)
for rz in rad_zones.filter(start=times.yday[model_idxs][0], stop=(times[-1]+2*u.day).yday):
    for ax in [ax1, ax2]:
        ax.axvspan(CxoTime(rz.tstart).datetime, 
                   CxoTime(rz.tstop).datetime,
                   color="mediumpurple", alpha=0.333333)

fig.savefig(out_path / "txings_proxy.png", bbox_inches='tight')
