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


def main(fn):
    
    with open(fn, "r") as f:
        inputs = json.load(f)
    
    trigger_names = {"acis": "ACIS TXings",
                     "hrc": "HRC Anti-Co Shield"}
    
    shutdown = inputs["shutdown_time"]
    startup = inputs["startup_time"]

    dt = TimeDelta(2.0*u.d)

    shutdown = CxoTime(shutdown)
    startup = CxoTime(startup)
    storm_date = shutdown.yday[:8]

    storm_dir = Path(f"doc/source/storm_{storm_date.replace(':','_')}")
    if not storm_dir.exists():
        storm_dir.mkdir()

    sw = storms.SolarWind(shutdown-dt, startup+dt)

    fig = sw.plot_all()
    for ax in fig.axes:
        ax.axvline(shutdown.plot_date, color='k')
        ax.axvline(startup.plot_date, color='k')
        ymin = ax.get_ylim()[0]
        ax.text(shutdown.plot_date + 0.1, 1.5*ymin, "SHUTDOWN", 
                fontsize=15, rotation="vertical")
        ax.text(startup.plot_date + 0.1, 1.5*ymin, "STARTUP", 
                fontsize=15, rotation="vertical")

    fig.savefig(storm_dir / "rad_vs_time.png")
    
    times = []
    for t in inputs["spectrum_times"]:
        times.append(CxoTime(t).secs)
    times.sort()
    fig = sw.plot_proton_spectra(times)
    fig.axes[1].axvline(shutdown.plot_date, color='k')
    fig.axes[1].axvline(startup.plot_date, color='k')
    ymin = fig.axes[1].get_ylim()[0]
    fig.axes[1].text((shutdown+TimeDelta(0.1*u.day)).plot_date, 1.5*ymin, "SHUTDOWN", 
                     rotation="vertical", fontsize=18)
    fig.axes[1].text((startup+TimeDelta(0.1*u.day)).plot_date, 1.5*ymin, "STARTUP", 
                     rotation="vertical", fontsize=18)
    xlim = [CxoTime(times[0]-0.5*86400.0).plot_date,
            CxoTime(times[-1]+0.5*86400.0).plot_date]
    fig.axes[1].set_xlim(*xlim)
    fig.savefig(storm_dir / "proton_spectra.png")

    fig = sw.scatter_plots()
    idx = np.searchsorted(sw.times, shutdown.secs)
    row = sw.table[idx]
    fig.axes[0].plot(row["p3"], row["hrc_shield"], 'x', mew=3, ms=20, 
                     color='C3', label="Shutdown")
    fig.axes[0].legend(fontsize=18)
    fig.axes[1].plot(row["p3"], row["de1"], 'x', mew=3, ms=20, color='C3')
    fig.savefig(storm_dir / "scatter_plots.png")

    storm_template_file = 'storm_template.rst'

    storm_template = open(Path("templates") / storm_template_file).read()
    storm_template = re.sub(r' %}\n', ' %}', storm_template)

    context = {"shutdown": inputs["shutdown"],
               "storm_date": storm_date,
               "load": inputs["load"],
               "trigger": trigger_names[inputs["trigger"]],
               "shutdown_time": shutdown.yday,
               "startup_time": startup.yday}

    outfile = storm_dir / "index.rst"

    template = jinja2.Template(storm_template)

    with open(outfile, "w") as f:
        f.write(template.render(**context))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fns = sys.argv[1:]
    else:
        p = Path(PurePath(__file__).parent / "json")
        fns = p.glob("*.json")
    for fn in fns:
        print(fn.name)
        main(fn)
