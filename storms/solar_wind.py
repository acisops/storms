import numpy as np
from astropy.table import Table
from astropy.time import TimeDelta
import astropy.units as u
from cxotime import CxoTime
from acispy.plots import CustomDatePlot, DummyDatePlot
from kadi.events import rad_zones
from ska_path import ska_path
from pathlib import Path
import cheta.fetch_sci as fetch
import matplotlib.pyplot as plt
from glob import glob
import h5py
from collections import defaultdict


ace_file_tmpl = "ACE_BROWSE_{}-*.h5"
arc_data = ska_path('data', 'arc')

ace_map = {
    "p1": "Ion_vlo_EPAM",
    "p3": "Ion_lo_EPAM",
    "p5": "Ion_mid_EPAM",
    "p7": "Ion_hi_EPAM",
    "de1": "e_lo_EPAM",
    "de4": "e_hi_EPAM"
}


class SolarWind:

    def __init__(self, start, stop):
        self.start = CxoTime(start)
        self.stop = CxoTime(stop)
        self.rad_zones = rad_zones.filter(self.start, self.stop)
        self._get_ace()
        self._get_hrc()

    def _get_ace(self):
        ace_data_dir = Path("/Users/jzuhone/ace")
        times = []
        table = defaultdict(list)
        for year in range(self.start.datetime.year, self.stop.datetime.year + 1):
            fn = glob(str(ace_data_dir / ace_file_tmpl.format(year)))[0]
            with h5py.File(fn, "r") as f:
                d = f['VG_ace_br_5min_avg']['ace_br_5min_avg'][()]
                tfile = CxoTime(d["fp_year"], format="frac_year")
                idxs = (tfile.secs >= self.start.secs) & (tfile.secs <= self.stop.secs)
                times.append(tfile.secs[idxs])
                for k, v in ace_map.items():
                    table[k].append(d[v][idxs])
        self.table = Table({k: np.concatenate(table[k]) for k in ace_map})
        self.times = CxoTime(np.concatenate(times), format='secs')
        # Replace data marked as bad or negative fluxes with NaNs
        for name in self.table.colnames:
            if name in ["de1", "de4"]:
                self.table[name][self.table[name] < 1.0e-3] = np.nan
            elif name in ["p1", "p3", "p5", "p7"]:
                self.table[name][self.table[name] < 1.0e-3] = np.nan
        self.e_ebins = np.array([[47, 65], [112, 187]])
        self.p_ebins = np.array([[46, 67], [112, 187], [310, 580],
                                 [1060, 1910]])

    def _get_hrc(self):
        hrc_data = Path(arc_data) / "hrc_shield.h5"
        with h5py.File(hrc_data, "r") as f:
            d = f["data"][()]
            t = d["time"]
            proxy = np.interp(self.times.secs, t, d["hrc_shield"])
        self.table["hrc_shield"] = proxy
        self.table["hrc_shield"][self.table["hrc_shield"] < 0.1] = np.nan
        msid = fetch.MSID("2shldart", self.start, self.stop)
        if len(msid.times) == 0:
            shield1 = np.zeros(self.times.size)
        else:
            msid.interpolate(times=self.times.secs)
            shield1 = msid.vals/256
        self.table["2shldart"] = shield1

    def __getitem__(self, item):
        return self.table[item]

    def plot_electrons(self):
        return self._plot_electrons()

    def _plot_electrons(self, plot=None):
        dp = CustomDatePlot(self.times, self["de1"], label="DE1", plot=plot)
        CustomDatePlot(self.times, self["de4"], plot=dp, label="DE4")
        dp.set_yscale("log")
        dp.set_ylabel("Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_legend(loc='upper left', fontsize=14)
        t = np.linspace(self.times.secs[0], self.times.secs[-1], 1000)
        tplot = CxoTime(t).plot_date
        for radzone in self.rad_zones:
            in_evt = (t >= radzone.tstart) & (t <= radzone.tstop)
            dp.ax.fill_between(tplot, 0, 1, where=in_evt,
                               transform=dp.ax.get_xaxis_transform(),
                               color="mediumpurple", alpha=0.333333)
        return dp

    def plot_protons(self):
        return self._plot_protons()

    def _plot_protons(self, plot=None):
        dp = CustomDatePlot(self.times, self["p1"], label="P1", plot=plot)
        CustomDatePlot(self.times, self["p3"], plot=dp, label="P3")
        CustomDatePlot(self.times, self["p5"], plot=dp, label="P5")
        CustomDatePlot(self.times, self["p7"], plot=dp, label="P7")
        dp.set_yscale("log")
        dp.set_ylabel("Differential Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_legend(loc='upper left', fontsize=14)
        t = np.linspace(self.times.secs[0], self.times.secs[-1], 1000)
        tplot = CxoTime(t).plot_date
        for radzone in self.rad_zones:
            in_evt = (t >= radzone.tstart) & (t <= radzone.tstop)
            dp.ax.fill_between(tplot, 0, 1, where=in_evt,
                               transform=dp.ax.get_xaxis_transform(),
                               color="mediumpurple", alpha=0.333333)
        return dp

    def plot_proton_spectra(self, times):
        spectrum_colors = ["tab:purple", "tab:brown", "tab:pink", 
                           "tab:gray", "tab:olive", "tab:cyan", "lime"]
        times = CxoTime(times)
        e = 0.5*self.p_ebins.sum(axis=1)
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
        plot = DummyDatePlot(fig, ax2)
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
        CustomDatePlot(self.times, self["2shldart"], plot=dp, label="HRC Shield Rate")
        dp.set_yscale("log")
        dp.set_ylabel("Counts")
        dp.set_legend(loc='upper left', fontsize=14)
        t = np.linspace(self.times.secs[0], self.times.secs[-1], 1000)
        tplot = CxoTime(t).plot_date
        for radzone in self.rad_zones:
            in_evt = (t >= radzone.tstart) & (t <= radzone.tstop)
            dp.ax.fill_between(tplot, 0, 1, where=in_evt,
                               transform=dp.ax.get_xaxis_transform(),
                               color="mediumpurple", alpha=0.333333)
        return dp

    def plot_all(self):
        dp1 = self._plot_protons()
        dp1.set_ylabel("ACE Proton Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp1.set_ylim(8, None)
        dp1.ax.tick_params(axis='x', direction='inout', which='major', bottom=True, top=True,
                           length=8, labeltop=True, labelbottom=False, labelsize=18)
        dp1.ax.tick_params(axis='x', direction='inout', which='minor', bottom=True, top=True,
                           length=4)
        dp1.ax.xaxis.set_label_position("top")
        for label in dp1.ax.get_xticklabels():
            label.set_rotation(-30)
            label.set_ha('right')
        dp2 = self._plot_electrons()
        dp2.set_ylabel("ACE Electron Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp2.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp2.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        dp2.set_ylim(8, None)
        dp2.ax.set_xlabel("")
        dp2.ax.set_xticklabels([])
        dp3 = self._plot_hrc()
        dp3.set_ylabel("HRC Shield Rate & Proxy (Counts)")
        dp3.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp3.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        for dp in [dp1, dp2, dp3]:
            dp.fig.tight_layout()
        return dp1, dp2, dp3

    def scatter_plots(self):
        fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), ncols=2)
        ax1.scatter(self["p3"], self["hrc_shield"])
        ax1.set_title("GOES HRC Proxy vs. ACE P3", fontsize=18)
        ax2.scatter(self["p3"], self["de1"])
        ax2.set_title("ACE DE1 vs. ACE P3", fontsize=18)
        for ax in [ax1, ax2]:
            ax.set_xlabel("ACE P3 Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)",
                          fontsize=18)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(which='major', width=2, length=6, labelsize=18)
            ax.tick_params(which='minor', width=2, length=3)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)
        ax1.set_ylabel("GOES Proxy (counts)", fontsize=18)
        ax2.set_ylabel("ACE DE1 Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)",
                       fontsize=18)
        return fig