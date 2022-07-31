import numpy as np
from astropy.table import Table
from astropy.time import TimeDelta
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


pswitch_time = CxoTime("2003:302:23:50:00").secs

ace_file_tmpl = "epam_data_5min_year{}.h5"
arc_data = ska_path('data', 'arc')

ace_channels = [
    "P1", "P3", "P5", "P7", "DE1", "DE4"
]

txings_keys = ["obsid", "times", "I0", "I1", "I2", "I3",
               "S0", "S1", "S2", "S3", "S4", "S5"]


class SolarWind:

    def __init__(self, start, stop):
        self.start = CxoTime(start)
        self.stop = CxoTime(stop)
        self.rad_zones = rad_zones.filter(self.start, self.stop)
        self.obsids = obsids.filter(self.start, self.stop)
        self.states = cmd_states.get_states(start, stop)
        self._get_ace()
        self._get_hrc()
        self._get_txings()

    def _get_ace(self):
        ace_data_dir = Path("/data/acis/ace")
        times = []
        table = defaultdict(list)
        for year in range(self.start.datetime.year, self.stop.datetime.year + 1):
            fn = ace_data_dir / ace_file_tmpl.format(year)
            with h5py.File(fn, "r") as f:
                d = f["VG_EPAM_data_5min"]["EPAM_data_5min"][()]
                tfile = CxoTime(d["fp_year"], format="frac_year")
                idxs = (tfile.secs >= self.start.secs) & (tfile.secs <= self.stop.secs)
                t = tfile.secs[idxs]
                times.append(t)
                for k in ace_channels:
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
        if self.start.secs < CxoTime("2020:237:03:45:19").secs:
            msid = fetch.MSID("2shldart", self.start, self.stop)
            if msid.times.size != 0:
                shield1 = np.interp(self.times.secs, msid.times, msid.vals/256)
                self.table["2shldart"] = shield1

    def _get_txings(self):
        self.txings_data = defaultdict(list)
        with h5py.File("/Users/jzuhone/xings.hdf5", "r") as f:
            for obsid in self.obsids:
                if obsid.obsid > 39999:
                    continue
                this_o = f["obsid"][()] == obsid.obsid
                if this_o.sum() > 0:
                    sidxs = (obsid.tstart < self.states["tstart"]) & (obsid.tstop > self.states["tstop"])
                    pow_cmd = self.states["power_cmd"][sidxs]
                    tst = self.states["tstart"][sidxs]
                    ccdc = self.states["ccd_count"][sidxs]
                    mode = f["processing_mode"].asstr()[this_o][0]
                    if mode.startswith("Te"):
                        ssc = "XTZ"
                    else:
                        ssc = "XCZ"
                    t0 = tst[np.char.count(pow_cmd, ssc) == 1][0]
                    ccd_count = ccdc[np.char.count(pow_cmd, ssc) == 1][0]
                    biast = (13.0+2*ccd_count)*60.0
                    for k in txings_keys:
                        v = f[k][this_o]
                        if k == "times":
                            v = np.float64(v-v[0])+t0+biast
                        self.txings_data[k].append(v)
        for i, times in enumerate(self.txings_data["times"]):
            n_fi = 0
            n_bi = 0
            fi_rate = np.zeros(self.txings_data["times"][i].shape)
            bi_rate = np.zeros(self.txings_data["times"][i].shape)
            for c in ["I0", "I1", "I2", "I3", "S0", "S2", "S4", "S5"]:
                if self.txings_data[c][i].all() > 0.0:
                    n_fi += 1
                    fi_rate += self.txings_data[c][i]
            for c in ["S1", "S3"]:
                if self.txings_data[c][i].all() > 0.0:
                    n_bi += 1
                    bi_rate += self.txings_data[c][i]
            if n_fi > 0:
                fi_rate /= n_fi
            if n_bi > 0:
                bi_rate /= n_bi
            self.txings_data["fi_rate"].append(fi_rate)
            self.txings_data["bi_rate"].append(bi_rate)

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
        h_all = self["hrc_shield"]
        if "2shldart" in self.table.colnames:
            CustomDatePlot(self.times, self["2shldart"], plot=dp, label="HRC Shield Rate")
            h_all = np.concatenate([h_all, self["2shldart"]])
        dp.set_yscale("log")
        dp.set_ylabel("Counts")
        dp.set_legend(loc='upper left', fontsize=14)
        dp.set_ylim(0.5*np.nanmin(h_all), 1.5*np.nanmax(h_all))
        self._plot_rzs(dp.ax)
        return dp

    def _plot_txings(self, fig, ax):
        for i, times in enumerate(self.txings_data["times"]):
            if i == 0:
                labels = ["FI", "BI"]
            else:
                labels = [None]*2
            _, _, _ = plot_cxctime(times, self.txings_data["fi_rate"][i], color="C0", 
                                   label=labels[0], fig=fig, ax=ax)
            _, _, _ = plot_cxctime(times, self.txings_data["bi_rate"][i], color="C1", 
                                   label=labels[1], fig=fig, ax=ax)
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
        plot1 = DummyDatePlot(fig, ax1)
        dp1 = self._plot_p3(plot=plot1)
        dp1.set_ylabel("ACE P3 Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp1.ax.tick_params(axis='x', direction='inout', which='major', bottom=True, top=True,
                           length=8, labeltop=True, labelbottom=False, labelsize=18)
        dp1.ax.tick_params(axis='x', direction='inout', which='minor', bottom=True, top=True,
                           length=4)
        self._plot_txings(fig, ax2)
        xlim = dp1.ax.get_xlim()
        ax2.set_xlim(*xlim)
        plot3 = DummyDatePlot(fig, ax3)
        dp3 = self._plot_hrc(plot=plot3)
        dp3.set_ylabel("HRC Shield Rate & Proxy\n(Counts)", fontsize=16)
        dp3.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp3.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        fig.tight_layout()
        return fig

    def plot_ace(self):
        fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), nrows=2)
        plot1 = DummyDatePlot(fig, ax1)
        dp1 = self._plot_protons(plot=plot1)
        dp1.set_ylabel("ACE Proton Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp1.ax.tick_params(axis='x', direction='inout', which='major', bottom=True, top=True,
                           length=8, labeltop=True, labelbottom=False, labelsize=18)
        dp1.ax.tick_params(axis='x', direction='inout', which='minor', bottom=True, top=True,
                           length=4)
        plot2 = DummyDatePlot(fig, ax2)
        dp2 = self._plot_electrons(plot=plot2)
        dp2.set_ylabel("ACE Electron Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=16)
        dp2.ax.tick_params(axis='x', direction='inout', which='major', length=8, top=True, bottom=True)
        dp2.ax.tick_params(axis='x', direction='inout', which='minor', length=4, top=True, bottom=True)
        dp2.ax.set_xlabel("")
        dp2.ax.set_xticklabels([])
        fig.tight_layout()
        return fig

    def scatter_plots(self):
        fig, (ax1, ax2) = plt.subplots(figsize=(20, 9.5), ncols=2)
        ax1.scatter(self["p3"], self["hrc_shield"])
        ax1.set_title("GOES HRC Proxy vs. ACE P3", fontsize=18)
        fi_rate = np.empty(self.times.secs.shape)
        fi_rate[:] = np.nan
        for i, times in enumerate(self.txings_data["times"]):
            idxs = np.searchsorted(self.times.secs, times)-1
            fi_rate[idxs] = self.txings_data["fi_rate"][i]
        ax2.scatter(self["hrc_shield"], fi_rate)
        ax2.set_title("FI Txings vs. Goes Proxy", fontsize=18)
        ax1.set_yscale("log")
        ax1.set_xlabel("ACE P3 Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)",
                      fontsize=18)
        ax2.set_xlabel("GOES Proxy (counts)", fontsize=18)
        for ax in [ax1, ax2]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(which='major', width=2, length=6, labelsize=18)
            ax.tick_params(which='minor', width=2, length=3, labelsize=18)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)
        ax1.set_ylabel("GOES Proxy (counts)", fontsize=18)
        ax2.set_ylabel("ACIS Threshold Crossing Rate (cts/row/s)", fontsize=18)
        return fig, fi_rate