#!/usr/bin/env python
# coding: utf-8

import storms
from cxotime import CxoTime, TimeDelta
import astropy.units as u
import numpy as np
import jinja2
import re
from pathlib import Path

trigger_names = {"acis": "ACIS TXings",
                 "hrc": "HRC Anti-Co Shield"}

#shutdown = "2022-03-28T12:40:00"
#startup = "2022-04-04T05:59:00"
shutdown = "2017-09-07T02:31:00"
startup = "2017-09-09T06:04:00"

dt = TimeDelta(2.0*u.d)

shutdown = CxoTime(shutdown)
startup = CxoTime(startup)
storm_date = shutdown.yday[:8]

storm_dir = Path(f"doc/source/storm_{storm_date.replace(':','_')}")
if not storm_dir.exists():
    storm_dir.mkdir()

sw = storms.SolarWind(shutdown-dt, startup+dt)

"""
dpe = sw.plot_electrons()
dpe.add_vline(shutdown, color='k')
dpe.add_vline(startup, color='k')
dpe.add_text(shutdown+TimeDelta(0.1*u.day), 1.0, "SHUTDOWN", rotation="vertical")
dpe.add_text(startup+TimeDelta(0.1*u.day), 1.0, "STARTUP", rotation="vertical")
dpe.savefig(storm_dir / "electrons_vs_time.png")

dpp = sw.plot_protons()
dpp.add_vline(shutdown, color='k')
dpp.add_vline(startup, color='k')
dpp.add_text(shutdown+TimeDelta(0.1*u.day), 1.0, "SHUTDOWN", rotation="vertical")
dpp.add_text(startup+TimeDelta(0.1*u.day), 1.0, "STARTUP", rotation="vertical")
dpp.savefig(storm_dir / "protons_vs_time.png")

gpp = sw.plot_hrc()
gpp.add_vline(shutdown, color='k')
gpp.add_vline(startup, color='k')
gpp.add_text(shutdown+TimeDelta(0.1*u.day), 1.0, "SHUTDOWN", rotation="vertical")
gpp.add_text(startup+TimeDelta(0.1*u.day), 1.0, "STARTUP", rotation="vertical")
gpp.savefig(storm_dir / "hrc_vs_time.png")
"""

dp1, dp2, dp3 = sw.plot_all()
for i, dp in enumerate([dp1, dp2, dp3]):
    dp.add_vline(shutdown, color='k')
    dp.add_vline(startup, color='k')
    ytxt = 2 if i == 2 else 20
    dp.add_text(shutdown + TimeDelta(0.1 * u.day), ytxt, "SHUTDOWN", rotation="vertical")
    dp.add_text(startup + TimeDelta(0.1 * u.day), ytxt, "STARTUP", rotation="vertical")

dp1.savefig(storm_dir / "protons_vs_time.png")
dp2.savefig(storm_dir / "electrons_vs_time.png")
dp3.savefig(storm_dir / "hrc_vs_time.png")

times = [(shutdown-(2-i)*TimeDelta(12*u.hr)).secs for i in range(4)]
times.append(CxoTime("2017:249:08:00:00").secs)
times.append(CxoTime("2017:249:23:00:00").secs)
times.sort()
fig = sw.plot_proton_spectra(times)
fig.axes[1].axvline(shutdown.plot_date, color='k')
fig.axes[1].axvline(startup.plot_date, color='k')
fig.axes[1].text((shutdown+TimeDelta(0.1*u.day)).plot_date, 15.0, "SHUTDOWN", 
                 rotation="vertical", fontsize=18)
fig.axes[1].text((startup+TimeDelta(0.1*u.day)).plot_date, 15.0, "STARTUP", 
                 rotation="vertical", fontsize=18)
fig.axes[1].set_ylim(10.0, None)
fig.savefig(storm_dir / "proton_spectra.png")

fig = sw.scatter_plots()
idx = np.searchsorted(sw.times, shutdown.secs)
row = sw.table[idx]
fig.axes[0].plot(row["p3"], row["hrc_shield"], 'x', mew=3, ms=20, color='C3', label="Shutdown")
fig.axes[0].legend(fontsize=18)
fig.axes[1].plot(row["p3"], row["de1"], 'x', mew=3, ms=20, color='C3')
fig.savefig(storm_dir / "scatter_plots.png")

storm_template_file = 'storm_template.rst'

storm_template = open(Path("templates") / storm_template_file).read()
storm_template = re.sub(r' %}\n', ' %}', storm_template)

context = {"shutdown": "YES",
           "storm_date": storm_date,
           "load": "SEP0417A",
           "trigger": trigger_names["hrc"],
           "shutdown_time": shutdown.yday,
           "startup_time": startup.yday}

outfile = storm_dir / "index.rst"

template = jinja2.Template(storm_template)

with open(outfile, "w") as f:
    f.write(template.render(**context))


