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
from astropy.table import Table


def plot_shutdown(sw, shutdown, startup, ax):
    ax.axvspan(shutdown.datetime, startup.datetime,
               color="C3", alpha=0.25, zorder=-10)
    ymin = ax.get_ylim()[0]
    ax.text(shutdown.plot_date + 0.1, 2.0 * ymin, "SHUTDOWN",
            fontsize=15, rotation="vertical")
    ax.text(startup.plot_date + 0.1, 2.0 * ymin, "STARTUP",
            fontsize=15, rotation="vertical")


def main(fn):

    with open(fn, "r") as f:
        inputs = json.load(f)

    skip = inputs.get("skip", False)
    if skip:
        return

    browse = inputs.get("browse", False)

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

    sw = storms.SolarWind(begin_time, end_time, browse=browse,
                          txings_files=inputs["txings_files"])

    fig, xlim = sw.plot_all()
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
        xlim2 = [CxoTime(times[0]-0.5*86400.0).plot_date,
                 CxoTime(times[-1]+0.5*86400.0).plot_date]
        fig.axes[1].set_xlim(*xlim2)
        fig.savefig(storm_dir / "proton_spectra.png")

    fig, fi_rate = sw.scatter_plots()
    if inputs["shutdown"] == "YES":
        p3_idx = np.searchsorted(sw.ace_times.secs, shutdown.secs)-1
        hrc_idx = np.searchsorted(sw.hrc_times.secs, shutdown.secs)-1
        fig.axes[0].plot(sw["p3"], sw["hrc_shield"], 'x', mew=3, ms=20,
                         color='C3', label="Shutdown")
        fig.axes[0].legend(fontsize=18)
        fig.axes[1].plot(row["hrc_shield"], fi_rate[idx], 'x', mew=3, ms=20, color='C3')
    fig.savefig(storm_dir / "scatter_plots.png")
    sw.table["ace_times"] = sw.ace_times.secs
    if sw.goes_r_times:
        sw.table["goes_r_times"] = sw.goes_r_times.secs
    if sw.hrc_times:
        sw.table["hrc_times"] = sw.hrc_times.secs
    sw.table.write(f"{str(fn)[:-4]}ecsv", format="ascii.ecsv", overwrite=True)

    txingst = Table(sw.rates)
    txingst.write(f"{str(fn)[:-5]}_txings.ecsv", format="ascii.ecsv", overwrite=True)

    fig = sw.plot_ace(xlim=xlim)
    if inputs["shutdown"] == "YES":
        for ax in fig.axes:
            plot_shutdown(sw, shutdown, startup, ax)

    fig.savefig(storm_dir / "ace_vs_time.png")

    dp = sw.plot_index()
    dp.ax.set_xlim(*xlim)
    dp.set_ylim(-6, None)
    if inputs["shutdown"] == "YES":
        plot_shutdown(sw, shutdown, startup, dp.ax)

    dp.savefig(storm_dir / "index_vs_time.png")

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
