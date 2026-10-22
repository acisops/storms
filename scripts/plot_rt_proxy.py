import argparse
from pathlib import Path

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from cxotime import CxoTime
from kadi.commands.states import get_states
from kadi.events import rad_zones
from ska_numpy import interpolate

from storms import SolarWind
from storms.txings_proxy.realtime import get_realtime_goes
from storms.txings_proxy.utils import prep_data, run_model

parser = argparse.ArgumentParser(description="Plot the txings GOES proxy.")

parser.add_argument("--out_path", type=str, help="The location to write the image.")
parser.add_argument(
    "--days", type=float, default=3.0, help="The number of days to show on the plot."
)
args = parser.parse_args()

out_path = Path(args.out_path if args.out_path else "/proj/web-cxc/htdocs/acis/")

plt.rc("font", size=18)
plt.rc("axes", linewidth=2)
plt.rc("xtick.major", size=6, width=2)
plt.rc("ytick.major", size=6, width=2)
plt.rc("xtick.minor", size=3, width=2)
plt.rc("ytick.minor", size=3, width=2)


t_proxy = get_realtime_goes(use7d=True)
X = prep_data(t_proxy)

for which_rate in ["fi_rate", "bi_rate"]:
    t_proxy[f"{which_rate}_predict"] = np.mean(run_model(X, which_rate), axis=0)

times = Time(t_proxy["time"], format="cxcsec")

tstop = times[-1]
tstart = times[-1] - args.days * u.day

print(tstart.yday, tstop.yday)

sw = SolarWind(tstart.yday, tstop.yday, get_txings=True, get_ace=False)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 13), constrained_layout=True)
fig_fi, ax_fi = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
fig_bi, ax_bi = plt.subplots(figsize=(12, 6.5), constrained_layout=True)

model_idxs = times >= tstop - args.days * u.day
model_times = times.datetime[model_idxs]
fi_rate_predict = t_proxy["fi_rate_predict"][model_idxs]
bi_rate_predict = t_proxy["bi_rate_predict"][model_idxs]

data_idxs = sw.txings_times >= tstop - args.days * u.day
data_times = sw.txings_times.datetime[data_idxs]
fi_rate = sw.txings_data["fi_rate"][data_idxs]
bi_rate = sw.txings_data["bi_rate"][data_idxs]
fi_rate_limit = interpolate(
    sw.txings_data["fi_rate_limit"][data_idxs],
    data_times,
    model_times,
    method="nearest",
)
bi_rate_limit = interpolate(
    sw.txings_data["bi_rate_limit"][data_idxs],
    data_times,
    model_times,
    method="nearest",
)


def plot_rate(ax, predict, limit, data, *, label, predict_color, data_color):
    ax.plot(
        model_times, predict, lw=3, label=f"{label} Prediction", color=predict_color
    )
    ax.plot(model_times, limit, label=f"{label} Limit", lw=2, color=predict_color)
    ax.plot(data_times, data, ".", ms=6, label=f"{label} Data", color=data_color)


for ax in [ax1, ax_fi]:
    plot_rate(
        ax,
        fi_rate_predict,
        fi_rate_limit,
        fi_rate,
        label="FI",
        predict_color="C0",
        data_color="C1",
    )

for ax in [ax2, ax_bi]:
    plot_rate(
        ax,
        bi_rate_predict,
        bi_rate_limit,
        bi_rate,
        label="BI",
        predict_color="C2",
        data_color="C3",
    )

fi_axes = [ax1, ax_fi]
bi_axes = [ax2, ax_bi]
all_axes = fi_axes + bi_axes

for ax in all_axes:
    ax.grid()
    ax.set_xlim(model_times[0], times[-1].datetime)
    ax.set_xlabel("Date")
    ax.legend(fontsize=16, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%j:%H"))

for f in [fig, fig_fi, fig_bi]:
    f.autofmt_xdate()

fi_ylim = (80, max(420.0, fi_rate.max() * 1.1, fi_rate_predict.max() * 1.1))
bi_ylim = (5, max(35.0, bi_rate.max() * 1.1, bi_rate_predict.max() * 1.1))

for ax in fi_axes:
    ax.set_ylim(*fi_ylim)
for ax in bi_axes:
    ax.set_ylim(*bi_ylim)

ylabel = "ACIS Threshold Crossing Rate (cts/sec/100 rows)"
fig.supylabel(ylabel)
ax_fi.set_ylabel(ylabel)
ax_bi.set_ylabel(ylabel)

states = get_states(
    start=times.yday[model_idxs][0], stop=times[-1].yday, merge_identical=True
)
oidxs = np.array(["obsid" in s for s in states["trans_keys"]])
oo = states["obsid"][oidxs]
tstart = CxoTime(states["tstart"][oidxs])

for o, t in zip(oo, tstart, strict=True):
    if t >= model_times[0] and o <= 39000:
        for ax in fi_axes:
            ax.text(
                t.datetime,
                100,
                str(o),
                color="C3",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=14,
                zorder=-100,
            )
        for ax in bi_axes:
            ax.text(
                t.datetime,
                8,
                str(o),
                color="C3",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=14,
                zorder=-100,
            )
for rz in rad_zones.filter(
    start=times.yday[model_idxs][0], stop=(times[-1] + 2 * u.day).yday
):
    for ax in all_axes:
        ax.axvspan(
            CxoTime(rz.tstart).datetime,
            CxoTime(rz.tstop).datetime,
            color="mediumpurple",
            alpha=0.333333,
        )

fig.savefig(out_path / "txings_proxy.png", bbox_inches="tight")
fig_fi.savefig(out_path / "txings_proxy_fi.png", bbox_inches="tight")
fig_bi.savefig(out_path / "txings_proxy_bi.png", bbox_inches="tight")
