import numpy as np
from astropy.table import Table, vstack
from astropy.time import TimeDelta
from astropy.io import ascii
import astropy.units as u
from cxotime import CxoTime
from acispy.plots import CustomDatePlot, DummyDatePlot
from kadi.events import rad_zones, obsids
from ska_path import ska_path
from pathlib import Path
import cheta.fetch_sci as fetch
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
from kadi.commands import states as cmd_states
from Ska.Matplotlib import plot_cxctime
from matplotlib import font_manager
from datetime import datetime
from calendar import isleap


def convert_txings_date(dates):
    times = []
    for d in dates:
        doy, secs = d.rsplit(":", 1)
        times.append((CxoTime(doy) + float(secs) * u.s).secs)
    return np.array(times)


pswitch_time = CxoTime("2003:302:23:50:00").secs

arc_data = ska_path('data', 'arc')

ace_channels = [
    "P1", "P3", "P5", "P7", "DE1", "DE4"
]

browse_channels = {
    "P1": "Ion_vlo_EPAM",
    "P3": "Ion_lo_EPAM",
    "P5": "Ion_mid_EPAM",
    "P7": "Ion_hi_EPAM",
    "DE1": "e_lo_EPAM",
    "DE4": "e_hi_EPAM",
}

txings_cols = [
    "processing_phase",
    "observing_run",
    "date",
    "obsid",
    "seconds",
    "triggered",
    "fi_rate_limit",
    "bi_rate_limit",
    "fi_rate",
    "bi_rate"
]

txings_keys = ["obsid", "times", "I0", "I1", "I2", "I3",
               "S0", "S1", "S2", "S3", "S4", "S5"]

goes_r_channels = ["p1", "p2a", "p2b", "p3", "p4", "p5",
                   "p6", "p7", "p8a", "p8b", "p8c", "p9", "p10"]

goes_r_bands = np.array([
    [1020, 1860],
    [1900, 2300],
    [2310, 3340],
    [3400, 6480],
    [5840, 11000],
    [11640, 23270],
    [25900, 38100],
    [40300, 73400],
    [83700, 98500],
    [99900, 118000],
    [115000, 143000],
    [160000, 242000],
    [276000, 404000]
])

e_ebins = np.array([[47, 65], [112, 187]])
p_ebins = np.array([[46, 67], [112, 187], [310, 580],
                    [1060, 1910]])


class SolarWind:

    def __init__(self, start, stop, browse=False, txings_files=None,
                 no_hrc=False):
        self.start = CxoTime(start)
        self.stop = CxoTime(stop)
        self.rad_zones = rad_zones.filter(self.start, self.stop)
        self.obsids = obsids.filter(self.start, self.stop)
        self.states = cmd_states.get_states(start, stop)
        self._get_ace(browse=browse)
        if not no_hrc:
            self._get_hrc()
        if txings_files:
            self._get_txings(txings_files)

    def _get_ace_file(self, year, browse=False):
        if browse:
            current_year = datetime.now().year
            if year == current_year:
                fn = f"ACE_BROWSE_{year}-001_to_current.h5"
            else:
                last_day = 366 if isleap(year) else 365
                fn = f"ACE_BROWSE_{year}-001_to_{year}-{last_day}.h5"
        else:
            fn = f"epam_data_5min_year{year}.h5"
        return fn

    def _get_ace(self, browse=False):
        ace_data_dir = Path("/data/acis/ace")
        times = []
        table = defaultdict(list)
        for year in range(self.start.datetime.year, self.stop.datetime.year + 1):
            fn = ace_data_dir / self._get_ace_file(year, browse=browse)
            with h5py.File(fn, "r") as f:
                if browse:
                    d = f['VG_ace_br_5min_avg']['ace_br_5min_avg'][()]
                else:
                    d = f["VG_EPAM_data_5min"]["EPAM_data_5min"][()]
                tfile = CxoTime(d["fp_year"], format="frac_year")
                idxs = (tfile.secs >= self.start.secs) & (tfile.secs <= self.stop.secs)
                t = tfile.secs[idxs]
                times.append(t)
                for k in ace_channels:
                    if browse:
                        v = d[browse_channels[k]][idxs]
                    else:
                        if k.startswith("P"):
                            v = d[k+"p"][idxs]
                            v[t < pswitch_time] = d[k][idxs][t < pswitch_time]
                        else:
                            v = d[k][idxs]
                    table[k].append(v)
        self.table = Table({k.lower(): np.concatenate(table[k]) for k in ace_channels})
        self.times = CxoTime(np.concatenate(times), format='secs')
        # Replace data marked as bad or negative fluxes with NaNs
        for name in self.table.colnames:
            if name in ["de1", "de4"]:
                self.table[name][self.table[name] < 1.0e-3] = np.nan
            elif name in ["p1", "p3", "p5", "p7"]:
                self.table[name][self.table[name] < 1.0e-3] = np.nan

    def _get_hrc(self):
        hrc_data = Path(arc_data) / "hrc_shield.h5"
        with h5py.File(hrc_data, "r") as f:
            d = f["data"][()]
            t = d["time"]
            self.table["hrc_shield"] = np.interp(self.times.secs, t, d["hrc_shield"])
            for pf in goes_r_channels:
                self.table[f"g{pf}"] = np.interp(self.times.secs, t, d[pf])
        bad = self.table["hrc_shield"] < 0.1
        self.table["hrc_shield"][bad] = np.nan
        for pf in goes_r_channels:
            self.table[f"g{pf}"][self.table[f"g{pf}"] <= 0.0] = np.nan
        if self.start.secs < CxoTime("2020:237:03:45:19").secs:
            msid = fetch.MSID("2shldart", self.start, self.stop)
            if msid.times.size != 0:
                shield1 = np.interp(self.times.secs, msid.times, msid.vals/256)
                self.table["2shldart"] = shield1

    def _get_txings(self, txings_files):
        t = vstack([ascii.read(f"/data/acis/txings/{fn}", names=txings_cols)
                    for fn in txings_files])
        times = convert_txings_date(t["date"].data)
        idxs = (times > self.start.secs) & (times < self.stop.secs)
        self.txings_data = t[idxs]
        self.txings_data["time"] = times[idxs]

    def __getitem__(self, item):
        return self.table[item]

    def _plot_rzs(self, ax):
        t = np.linspace(self.times.secs[0], self.times.secs[-1], 1000)
        tplot = CxoTime(t).plot_date
        for radzone in self.rad_zones:
            in_evt = (t >= radzone.tstart) & (t <= radzone.tstop)
            ax.fill_between(tplot, 0, 1, where=in_evt,
                            transform=ax.get_xaxis_transform(),
                            color="mediumpurple", alpha=0.333333)

    def plot_electrons(self):
        return self._plot_electrons()

    def _plot_electrons(self, plot=None):
        dp = CustomDatePlot(self.times, self["de1"], label="DE1", plot=plot)
        CustomDatePlot(self.times, self["de4"], plot=dp, label="DE4")
        de_all = np.concatenate([self[f"de{i}"] for i in [1, 4]])
        dp.set_yscale("log")
        dp.set_ylabel("Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_legend(loc='upper left', fontsize=14)
        dp.set_ylim(0.5*np.nanmin(de_all), 1.5*np.nanmax(de_all))
        self._plot_rzs(dp.ax)
        return dp

    def plot_protons(self):
        return self._plot_protons()

    def _plot_p3(self, plot=None):
        dp = CustomDatePlot(self.times, self["p3"], plot=plot, color="C1")
        dp.set_yscale("log")
        dp.set_ylabel("ACE P3 Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_ylim(0.5*np.nanmin(self["p3"]), 1.5*np.nanmax(self["p3"]))
        self._plot_rzs(dp.ax)
        return dp

    def _plot_protons(self, plot=None):
        dp = CustomDatePlot(self.times, self["p1"], label="P1", plot=plot)
        CustomDatePlot(self.times, self["p3"], plot=dp, label="P3")
        CustomDatePlot(self.times, self["p5"], plot=dp, label="P5")
        CustomDatePlot(self.times, self["p7"], plot=dp, label="P7")
        p_all = np.concatenate([self[f"p{i}"] for i in [1, 3, 5, 7]])
        dp.set_yscale("log")
        dp.set_ylabel("Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_legend(loc='upper left', fontsize=14)
        dp.set_ylim(0.5*np.nanmin(p_all), 1.5*np.nanmax(p_all))
        self._plot_rzs(dp.ax)
        return dp

    def _plot_goes_r(self, plot=None):
        dp = CustomDatePlot(self.times, self["gp1"], label="P1", plot=plot)
        mins = [np.nanmin(self["gp1"])]
        maxes = [np.nanmax(self["gp1"])]
        for pf in goes_r_channels[1:8]:
            CustomDatePlot(self.times, self[f"g{pf}"], plot=dp, label=pf.upper())
            mins.append(np.nanmin(self[f"g{pf}"]))
            maxes.append(np.nanmax(self[f"g{pf}"]))
        dp.set_yscale("log")
        dp.set_ylabel("Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_legend(loc='upper left', fontsize=14)
        ymin = np.nanmin(mins)
        ymax = np.nanmin(maxes)
        if ymin is np.nan or ymax is np.nan:
            return None
        else:
            dp.set_ylim(0.5*np.nanmin(mins), 1.5*np.nanmax(maxes))
            self._plot_rzs(dp.ax)
            return dp

    def plot_proton_spectra(self, times):
        spectrum_colors = ["tab:purple", "tab:brown", "tab:pink",
                           "tab:gray", "tab:olive", "tab:cyan", "lime"]
        times = CxoTime(times)
        e = 0.5*p_ebins.sum(axis=1)
        fig, (ax1, ax2) = plt.subplots(figsize=(20,10), ncols=2)
        for i, time in enumerate(times):
            idx = np.searchsorted(self.times.secs, time.secs)
            row = self.table[idx]
            ax1.plot(e, [row["p1"], row["p3"], row["p5"], row["p7"]], 'x-', lw=2,
                     label=time.yday, color=spectrum_colors[i], markersize=10, mew=3)
        ax1.legend(fontsize=16)
        ax1.set_xlabel("E (keV)", fontsize=18)
        ax1.set_ylabel("Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", 
                        fontsize=18)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.tick_params(which='major', width=2, length=6, labelsize=18)
        ax1.tick_params(which='minor', width=2, length=3)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(2)
        plot = DummyDatePlot(fig, ax2, [], None, [])
        dp = self._plot_protons(plot=plot)
        for i, time in enumerate(times):
            dp.add_vline(time, color=spectrum_colors[i], ls='--', lw=5)
        dt = TimeDelta(1.0*u.d)
        dp.set_xlim(CxoTime(times[0]) - 0.1*dt,
                    CxoTime(times[-1]) + 0.1*dt)
        ax3 = ax1.twiny()
        ax3.set_xlim(*ax1.get_xlim())
        ax3.set_xscale('log')
        ax3.set_xticks(e)
        ax3.set_xticklabels(["P1", "P3", "P5", "P7"])
        ax3.tick_params(which='major', width=2, length=6, labelsize=18)
        return fig

    def plot_hrc(self):
        return self._plot_hrc()

    def _plot_hrc(self, plot=None):
        dp = CustomDatePlot(self.times, self["hrc_shield"], label="GOES proxy", plot=plot)
        h_all = self["hrc_shield"]
        if "2shldart" in self.table.colnames:
            CustomDatePlot(self.times, self["2shldart"], plot=dp, label="HRC Shield Rate")
            h_all = np.concatenate([h_all, self["2shldart"]])
        dp.set_yscale("log")
        dp.set_ylabel("Counts")
        dp.set_legend(loc='upper left', fontsize=14)
        dp.set_ylim(0.5*np.nanmin(h_all), 250.0)
        dp.add_hline(235.0, ls='--', lw=2, color="C3")
        self._plot_rzs(dp.ax)
        return dp

    def _plot_txings(self, fig, ax):
        get_labels = True
        for i, o in enumerate(self.obsids):
            if o.obsid > 39999:
                continue
            if get_labels:
                labels = ["FI", "BI", "FI Limit", "BI Limit"]
                get_labels = False
            else:
                labels = [None]*4
            this_o = self.txings_data["obsid"] == o.obsid
            t = self.txings_data[this_o]
            _, _, _ = plot_cxctime(t["time"], t["fi_rate"]*0.01, fmt='.', color="C0",
                                   ls='', label=labels[0], fig=fig, ax=ax)
            _, _, _ = plot_cxctime(t["time"], t["bi_rate"]*0.01, fmt='.', color="C1",
                                   ls='', label=labels[1], fig=fig, ax=ax)
            _, _, _ = plot_cxctime(t["time"], t["fi_rate_limit"]*0.01, color="C0",
                                   label=labels[2], fig=fig, ax=ax)
            _, _, _ = plot_cxctime(t["time"], t["bi_rate_limit"]*0.01, color="C1",
                                   label=labels[3], fig=fig, ax=ax)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 100)
        ax.set_ylabel("ACIS Threshold Crossing Rate (cts/row/s)", fontsize=18)
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        fontProperties = font_manager.FontProperties(size=18)
        for label in ax.get_xticklabels():
            label.set_fontproperties(fontProperties)
        for label in ax.get_yticklabels():
            label.set_fontproperties(fontProperties)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.legend(fontsize=18)
        self._plot_rzs(ax)

    def plot_all(self):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10,15), nrows=3)
        plot1 = DummyDatePlot(fig, ax1, [], None, [])
        dp1 = self._plot_p3(plot=plot1)
        dp1.set_ylabel("ACE P3 Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp1.ax.tick_params(axis='x', direction='inout', which='major', bottom=True, top=True,
                           length=8, labeltop=True, labelbottom=False, labelsize=18)
        dp1.ax.tick_params(axis='x', direction='inout', which='minor', bottom=True, top=True,
                           length=4)
        self._plot_txings(fig, ax2)
        ax2.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        ax2.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        xlim = dp1.ax.get_xlim()
        ax2.set_xlim(*xlim)
        plot3 = DummyDatePlot(fig, ax3, [], None, [])
        dp3 = self._plot_hrc(plot=plot3)
        dp3.set_ylabel("HRC Shield Rate & Proxy\n(Counts)", fontsize=16)
        dp3.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp3.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        ax3.set_xlim(*xlim)
        fig.tight_layout()
        return fig, xlim

    def plot_ace(self, xlim=None):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 15), nrows=3)
        plot1 = DummyDatePlot(fig, ax1, [], None, [])
        dp1 = self._plot_protons(plot=plot1)
        dp1.set_ylabel("ACE Proton Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp1.ax.tick_params(axis='x', direction='inout', which='major', bottom=True, top=True,
                           length=8, labeltop=True, labelbottom=False, labelsize=18)
        dp1.ax.tick_params(axis='x', direction='inout', which='minor', bottom=True, top=True,
                           length=4)
        plot2 = DummyDatePlot(fig, ax2, [], None, [])
        dp2 = self._plot_electrons(plot=plot2)
        dp2.set_ylabel("ACE Electron Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp2.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp2.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        dp2.ax.set_xlabel("")
        dp2.ax.set_xticklabels([])
        plot3 = DummyDatePlot(fig, ax3, [], None, [])
        dp3 = self._plot_goes_r(plot=plot3)
        dp3.set_ylabel("GOES Proton Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp3.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp3.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        dp3.set_ylim(1.0e-6, None)
        if xlim is not None:
            ax1.set_xlim(*xlim)
            ax2.set_xlim(*xlim)
            ax3.set_xlim(*xlim)
        fig.tight_layout()
        return fig

    def plot_goes_r(self, xlim=None):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        plot1 = DummyDatePlot(fig, ax, [], None, [])
        dp1 = self._plot_goes_r(plot=plot1)
        dp1.set_ylabel("GOES Proton Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp1.ax.tick_params(axis='x', direction='inout', which='major', bottom=True, top=True,
                           length=8, labelsize=18)
        dp1.ax.tick_params(axis='x', direction='inout', which='minor', bottom=True, top=True,
                           length=4, labelsize=18)
        if xlim is not None:
            ax.set_xlim(*xlim)
        return fig 

    def scatter_plots(self):
        fig, (ax1, ax2) = plt.subplots(figsize=(20, 9.5), ncols=2)
        ax1.scatter(self["p3"], self["hrc_shield"])
        ax1.set_title("GOES HRC Proxy vs. ACE P3", fontsize=18)
        ax1.set_xlabel("ACE P3 Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)",
                      fontsize=18)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        fi_rate = np.empty(self.times.secs.shape)
        fi_rate[:] = np.nan
        for o in self.obsids:
            if o.obsid > 39999:
                continue
            this_o = self.txings_data["obsid"] == o.obsid
            t = self.txings_data[this_o]
            idxs = np.searchsorted(self.times.secs, t["time"])-1
            fi_rate[idxs] = t["fi_rate"]*0.01
        ax2.scatter(self["hrc_shield"], fi_rate)
        ax2.set_title("FI Txings vs. Goes Proxy", fontsize=18)
        ax2.set_xlabel("GOES Proxy (counts)", fontsize=18)
        ax2.set_xscale('log')
        if np.all(np.nan_to_num(fi_rate) > 0):
            ax2.set_yscale('log')
        for ax in [ax1, ax2]:
            ax.tick_params(which='major', width=2, length=6, labelsize=18)
            ax.tick_params(which='minor', width=2, length=3, labelsize=18)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)
        ax1.set_ylabel("GOES Proxy (counts)", fontsize=18)
        ax2.set_ylabel("ACIS Threshold Crossing Rate (cts/row/s)", fontsize=18)
        return fig, fi_rate