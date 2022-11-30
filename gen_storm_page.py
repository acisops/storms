#!/usr/bin/env python
# coding: utf-8

import storms
from cxotime import CxoTime, TimeDelta
import astropy.units as u
import numpy as np
import jinja2
import re
from pathlib import Path, PurePath
import json
import sys


def plot_shutdown(sw, shutdown, startup, ax):
    t = np.linspace(sw.times.secs[0], sw.times.secs[-1], 1000)
    tplot = CxoTime(t).plot_date
    in_evt = (t >= shutdown.secs) & (t <= startup.secs)
    ax.fill_between(tplot, 0, 1, where=in_evt,
                    transform=ax.get_xaxis_transform(),
                    color="C3", alpha=0.25, zorder=-10)
    ymin = ax.get_ylim()[0]
    ax.text(shutdown.plot_date + 0.1, 1.5 * ymin, "SHUTDOWN",
            fontsize=15, rotation="vertical")
    ax.text(startup.plot_date + 0.1, 1.5 * ymin, "STARTUP",
            fontsize=15, rotation="vertical")


def main(fn):
    
    with open(fn, "r") as f:
        inputs = json.load(f)

    trigger_names = {"acis": "ACIS TXings",
                     "hrc": "HRC Anti-Co Shield",
                     "manual": "Manual"}

    if inputs["shutdown"] == "YES":
        shutdown = CxoTime(inputs["shutdown_time"])
        startup = CxoTime(inputs["startup_time"])
        dt = TimeDelta(2.0*u.d)
        begin_time = shutdown-dt
        end_time = startup+dt
        storm_date = shutdown.yday[:8]
    else:
        shutdown = None
        startup = None
        begin_time = CxoTime(inputs["begin_time"])
        end_time = CxoTime(inputs["end_time"])
        storm_date = begin_time.yday[:8]

    storm_dir = Path(f"doc/source/storm_{storm_date.replace(':','_')}")
    if not storm_dir.exists():
        storm_dir.mkdir()

    sw = storms.SolarWind(begin_time, end_time)

    fig = sw.plot_all()
    if inputs["shutdown"] == "YES":
        for ax in fig.axes:
            plot_shutdown(sw, shutdown, startup, ax)

    fig.savefig(storm_dir / "rad_vs_time.png")

    times = []
    for t in inputs["spectrum_times"]:
        times.append(CxoTime(t).secs)
    times.sort()
    if len(times) > 0:
        fig = sw.plot_proton_spectra(times)
        if inputs["shutdown"] == "YES":
            plot_shutdown(sw, shutdown, startup, fig.axes[1])
        xlim = [CxoTime(times[0]-0.5*86400.0).plot_date,
                CxoTime(times[-1]+0.5*86400.0).plot_date]
        fig.axes[1].set_xlim(*xlim)
        fig.savefig(storm_dir / "proton_spectra.png")

    fig, fi_rate = sw.scatter_plots()
    if inputs["shutdown"] == "YES":
        idx = np.searchsorted(sw.times, shutdown.secs)-1
        row = sw.table[idx]
        fig.axes[0].plot(row["p3"], row["hrc_shield"], 'x', mew=3, ms=20,
                         color='C3', label="Shutdown")
        fig.axes[0].legend(fontsize=18)
        fig.axes[1].plot(row["hrc_shield"], fi_rate[idx], 'x', mew=3, ms=20, color='C3')
    fig.savefig(storm_dir / "scatter_plots.png")

    fig = sw.plot_ace()
    if inputs["shutdown"] == "YES":
        for ax in fig.axes:
            plot_shutdown(sw, shutdown, startup, ax)

    fig.savefig(storm_dir / "ace_vs_time.png")

    storm_template_file = 'storm_template.rst'

    storm_template = open(Path("templates") / storm_template_file).read()
    storm_template = re.sub(r' %}\n', ' %}', storm_template)

    context = {"shutdown": inputs["shutdown"],
               "storm_date": storm_date,
               "load": inputs["load"]}
    if inputs["shutdown"] == "YES":
        context["trigger"] = trigger_names[inputs["trigger"]]
        context["shutdown_time"] = shutdown.yday
        context["startup_time"] = startup.yday

    outfile = storm_dir / "index.rst"

    template = jinja2.Template(storm_template)

    with open(outfile, "w") as f:
        f.write(template.render(**context))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fns = [Path(sys.argv[1])]
    else:
        p = Path(PurePath(__file__).parent / "json")
        fns = p.glob("*.json")
    for fn in fns:
        print(fn.name)
        main(fn)
