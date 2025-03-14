import acispy
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
from more_itertools import always_iterable


def convert_txings_date(dates):
    dates = always_iterable(dates)
    doy = []
    secs = []
    for d in dates:
        d1, d2 = d.rsplit(":", 1)
        doy.append(d1)
        secs.append(float(d2))
    times = CxoTime(doy) + secs * u.s
    return times.secs


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
    "Plo": "H_lo_SIS",
    "Phi": "H_hi_SIS",
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

goes_r_channels = ["P1", "P2A", "P2B", "P3", "P4", "P5",
                   "P6", "P7", "P8A", "P8B", "P8C", "P9", "P10"]

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

goes_bands = np.array([
    [0.74, 4.2],
    [4.2, 8.7],
    [8.7, 14.5],
    [15.0, 40.0],
    [38, 82],
    [84, 200],
    [110, 900]
])*1.0e3

e_goes = 0.5*goes_bands.sum(axis=1)
e_goes_r = 0.5*goes_r_bands.sum(axis=1)

e_ebins = np.array([[47, 65], [112, 187]])
e_emid = 0.5 * e_ebins.sum(axis=1)
p_ebins = np.array([[46, 67], [112, 187], [310, 580],
                    [1060, 1910]])
p_emid = 0.5 * p_ebins.sum(axis=1)


ace_data_dir = Path("/data/acis/ace")
goes_data_dir = Path("/data/acis/goes")
txings_data_dir = Path("/data/acis/txings")


def find_index(t, e, channels):
    loge = np.log10(e)
    X = np.vstack([np.ones(e.size), loge]).T
    logF = np.log10([t[c] for c in channels])
    beta = np.linalg.inv(np.dot(X.T,X)) @ np.dot(X.T, logF)
    return beta[1]


def get_comm_times(start, stop):
    from kadi.events import dsn_comms
    comm_times = []
    comms = dsn_comms.filter(start=start, stop=stop)
    for comm in comms:
        words = comm.start.split(":")
        words[2] = comm.bot[:2]
        words[3] = comm.bot[2:]
        comm_start = CxoTime(":".join(words))
        if comm_start.secs < comm.tstart:
            comm_start += 1 * u.day
        words = comm.stop.split(":")
        words[2] = comm.eot[:2]
        words[3] = comm.eot[2:]
        comm_stop = CxoTime(":".join(words))
        if comm_stop.secs > comm.tstop:
            comm_stop -= 1 * u.day
        comm_times.append([comm_start.yday, comm_stop.yday])
    return comm_times


class SolarWind:

    def __init__(self, start, stop, use_browse=False, get_txings=False,
                 get_goes=False, get_hrc=False, get_states=False, 
                 get_comms=True):
        self.start = CxoTime(start)
        self.stop = CxoTime(stop)
        self.rad_zones = rad_zones.filter(self.start, self.stop)
        self.obsids = obsids.filter(self.start, self.stop)
        self._get_ace(use_browse=use_browse)
        if get_hrc:
            self._get_hrc()
        else:
            self.hrc_times = None
        if get_txings:
            self._get_txings()
        else:
            self.txings_times = None
        if get_goes:
            self._get_goes_r()
        else:
            self.goes_r_times = None
        if get_states:
            self.states = cmd_states.get_states(start, stop)
        else:
            self.states = None
        if get_comms:
            self.comms = get_comm_times(start, stop)
        else:
            self.comms = None

    def _get_ace_file(self, year, use_browse=False):
        if use_browse:
            current_year = datetime.now().year
            if year == current_year:
                fn = f"ACE_BROWSE_{year}-001_to_current.h5"
            else:
                last_day = 366 if isleap(year) else 365
                fn = f"ACE_BROWSE_{year}-001_to_{year}-{last_day}.h5"
        else:
            fn = f"epam_data_5min_year{year}.h5"
        fn = ace_data_dir / fn
        if not fn.exists():
            # sometimes we end up in a spot where they add data from a new
            # year to the end of the last file, so try it here
            fn = Path(str(fn).replace(str(year), str(year-1)))
        return fn

    def _get_ace(self, use_browse=False):
        times = []
        table = defaultdict(list)
        for year in range(self.start.datetime.year, self.stop.datetime.year + 1):
            fn = ace_data_dir / self._get_ace_file(year, use_browse=use_browse)
            if not fn.exists():
                # sometimes we end up in a spot where they add data from a new
                # year to the end of the last file, so try it here
                fn = ace_data_dir / self._get_ace_file(year-1, use_browse=use_browse)
            with h5py.File(fn, "r") as f:
                if use_browse:
                    d = f['VG_ace_br_5min_avg']['ace_br_5min_avg'][()]
                else:
                    d = f["VG_EPAM_data_5min"]["EPAM_data_5min"][()]
                tfile = CxoTime(d["fp_year"], format="frac_year")
                idxs = (tfile.secs >= self.start.secs) & (tfile.secs <= self.stop.secs)
                t = tfile.secs[idxs]
                times.append(t)
                for k in ace_channels:
                    if use_browse:
                        v = d[browse_channels[k]][idxs]
                    else:
                        if k.startswith("P"):
                            v = d[k+"p"][idxs]
                            v[t < pswitch_time] = d[k][idxs][t < pswitch_time]
                        else:
                            v = d[k][idxs]
                    table[k].append(v)
        self.ace_table = Table({k.lower(): np.concatenate(table[k]) for k in ace_channels})
        self.ace_times = CxoTime(np.concatenate(times), format='secs')
        # Replace data marked as bad or negative fluxes with NaNs
        for name in self.ace_table.colnames:
            self.ace_table[name][self.ace_table[name] < 1.0e-3] = np.nan

    def _get_hrc(self):
        self.hrc_table = {}
        hrc_data = Path(arc_data) / "hrc_shield.h5"
        with h5py.File(hrc_data, "r") as f:
            d = f["data"][()]
            t = d["time"]
        idxs = (t > self.start.secs) & (t < self.stop.secs)
        self.hrc_times = CxoTime(t[idxs], format='secs')
        self.hrc_table["hrc_shield"] = d["hrc_shield"][idxs]
        bad = self.hrc_table["hrc_shield"] < 0.1
        self.hrc_table["hrc_shield"][bad] = np.nan
        msids = fetch.MSIDset(["2shldart", "2shldbrt"], self.start, self.stop)
        if msids["2shldart"].times.size != 0:
            shield1 = np.interp(self.hrc_times.secs, msids["2shldart"].times, msids["2shldart"].vals/256)
            self.hrc_table["2shldart"] = shield1
        if msids["2shldbrt"].times.size != 0:
            shield2 = np.interp(self.hrc_times.secs, msids["2shldbrt"].times, msids["2shldbrt"].vals/256)
            self.hrc_table["2shldbrt"] = shield2
    
    def _get_goes_r(self):
        t = Table.read("/data/acis/goes/goes_16_18.fits", format="fits")
        times = t["time"].data
        idxs = (times > self.start.secs) & (times < self.stop.secs)
        self.goes_r_times = CxoTime(times[idxs], format='secs')
        self.goes_r_table = Table()
        for k in goes_r_channels:
            self.goes_r_table[k] = t[f"{k}_g16"][idxs,0]
            
    def _find_txings_files(self):
        txings_files = []
        fns = [str(fn) for fn in Path(txings_data_dir).glob("acis*.txt")]
        fns.sort(key=lambda x:f"{x[4:].split('.')[0].zfill(4)}")
        for fn in fns:
            with open(fn, "r") as f:
                lines = list(f.readlines())
                tstartf, tstopf = convert_txings_date([lines[0].split()[2],
                                                       lines[-1].split()[2]])
                if tstartf < self.stop.secs and tstopf > self.start.secs:
                    txings_files.append(fn)
        return txings_files
    
    def _get_txings(self):
        txings_files = self._find_txings_files()
        t = vstack([ascii.read(fn, names=txings_cols)
                    for fn in txings_files])
        times = convert_txings_date(t["date"].data)
        idxs = (times > self.start.secs) & (times < self.stop.secs)
        self.txings_data = t[idxs]
        self.txings_times = CxoTime(times[idxs], format='secs')

    def generate_slopes(self):
        self.ace_table["ace_soft_slope"] = find_index(self, p_emid[:-1],
                                             ["p1", "p3", "p5"])
        self.ace_table["ace_hard_slope"] = find_index(self, p_emid[1:],
                                             ["p3", "p5", "p7"])
        self.hrc_table["goes_soft_slope"] = find_index(self, e_goes_r[:4],
                                              ["gp1", "gp2a", "gp2b", "gp3"])

    def _plot_comms(self, ax):
        if self.comms:
            for i, comm in enumerate(self.comms):
                label = "Comm" if i == 0 else None
                ax.axvspan(CxoTime(comm[0]).datetime, CxoTime(comm[1]).datetime,
                           color="dodgerblue", alpha=0.5, label=label)

    def _plot_rzs(self, ax):
        for i, radzone in enumerate(self.rad_zones):
            label = "Rad Zone" if i == 0 else None
            ax.axvspan(CxoTime(radzone.tstart).datetime, 
                       CxoTime(radzone.tstop).datetime,
                       color="mediumpurple", label=label, alpha=0.333333)

    def plot_ace_e(self):
        return self._plot_ace_e()

    def _plot_ace_e(self, plot=None):
        dp = CustomDatePlot(self.ace_times, self.ace_table["de1"], label="DE1", plot=plot)
        CustomDatePlot(self.ace_times, self.ace_table["de4"], plot=dp, label="DE4")
        de_all = np.concatenate([self.ace_table[f"de{i}"] for i in [1, 4]])
        dp.set_yscale("log")
        dp.set_ylabel("ACE Electron Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_ylim(0.5*np.nanmin(de_all), 1.5*np.nanmax(de_all))
        self._plot_rzs(dp.ax)
        self._plot_comms(dp.ax)
        dp.set_legend(loc='upper left', fontsize=14)
        return dp

    def plot_ace_p(self):
        return self._plot_ace_p()

    def _plot_ace_p3(self, plot=None):
        dp = CustomDatePlot(self.ace_times, self.ace_table["p3"], plot=plot, color="C1", 
                            label="P3")
        dp.set_yscale("log")
        dp.set_ylabel("ACE P3 Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_ylim(0.5*np.nanmin(self.ace_table["p3"]), 1.5*np.nanmax(self.ace_table["p3"]))
        self._plot_rzs(dp.ax)
        self._plot_comms(dp.ax)
        dp.set_legend(loc='upper left', fontsize=14)
        return dp

    def plot_ace_p3(self):
        return self._plot_ace_p3()

    def _plot_ace_p(self, plot=None):
        dp = CustomDatePlot(self.ace_times, self.ace_table["p1"], label="P1", plot=plot)
        CustomDatePlot(self.ace_times, self.ace_table["p3"], plot=dp, label="P3")
        CustomDatePlot(self.ace_times, self.ace_table["p5"], plot=dp, label="P5")
        CustomDatePlot(self.ace_times, self.ace_table["p7"], plot=dp, label="P7")
        p_all = np.concatenate([self.ace_table[f"p{i}"] for i in [1, 3, 5, 7]])
        dp.set_yscale("log")
        dp.set_ylabel("ACE Proton Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        dp.set_ylim(0.5*np.nanmin(p_all), 1.5*np.nanmax(p_all))
        self._plot_rzs(dp.ax)
        self._plot_comms(dp.ax)
        dp.set_legend(loc='upper left', fontsize=14)
        return dp

    def _plot_goes_r(self, plot=None):
        dp = CustomDatePlot(self.goes_r_times, self.goes_r_table["P1"], 
                            label="P1", plot=plot)
        mins = [np.nanmin(self.goes_r_table["P1"])]
        maxes = [np.nanmax(self.goes_r_table["P1"])]
        for pf in ["P3", "P5", "P7"]:
            CustomDatePlot(self.goes_r_times, self.goes_r_table[pf], 
                           plot=dp, label=pf.upper())
            mins.append(np.nanmin(self.goes_r_table[pf]))
            maxes.append(np.nanmax(self.goes_r_table[pf]))
        dp.set_yscale("log")
        dp.set_ylabel("GOES Proton Flux\n(particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
        ymin = np.nanmin(mins)
        ymax = np.nanmin(maxes)
        if ymin is np.nan or ymax is np.nan:
            return None
        else:
            dp.set_ylim(0.5*np.nanmin(mins), 1.5*np.nanmax(maxes))
            self._plot_rzs(dp.ax)
            self._plot_comms(dp.ax)
            dp.set_legend(loc='upper left', fontsize=14)
            return dp

    def _plot_index(self, plot=None):
        if "ace_soft_slope" not in self.ace_table.colnames:
            self.generate_slopes()
        dp = CustomDatePlot(self.ace_times, self.ace_table["ace_soft_slope"], label="ACE P1-P5", plot=plot)
        CustomDatePlot(self.ace_times, self.ace_table["ace_hard_slope"], plot=dp, label="ACE P3-P7")
        CustomDatePlot(self.goes_r_times, self.goes_r_table["goes_soft_slope"], plot=dp, label="GOES P1-P3")
        dp.set_ylabel("Spectral Index")
        self._plot_rzs(dp.ax)
        self._plot_comms(dp.ax)
        dp.set_legend()
        return dp 

    def plot_index(self):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        plot1 = DummyDatePlot(fig, ax, [], None, [])
        return self._plot_index(plot=plot1)

    def plot_proton_spectra(self, times):
        spectrum_colors = ["tab:purple", "tab:brown", "tab:pink",
                           "tab:gray", "tab:olive", "tab:cyan", "lime"]
        times = CxoTime(times)
        fig, (ax1, ax2) = plt.subplots(figsize=(20,10), ncols=2)
        for i, time in enumerate(times):
            idx = np.searchsorted(self.ace_times.secs, time.secs)
            row = self.ace_table[idx]
            ax1.plot(p_emid, [row["p1"], row["p3"], row["p5"], row["p7"]], 'x-', lw=2,
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
        ax3.set_xticks(p_emid)
        ax3.set_xticklabels(["P1", "P3", "P5", "P7"])
        ax3.tick_params(which='major', width=2, length=6, labelsize=18)
        return fig

    def plot_hrc(self):
        return self._plot_hrc()

    def _plot_hrc(self, plot=None):
        dp = CustomDatePlot(self.hrc_times, self.hrc_table["hrc_shield"], label="GOES proxy", plot=plot)
        h_all = self.hrc_table["hrc_shield"]
        #if "2shldart" in self.hrc_table and np.nansum(self.hrc_table["2shldart"]) > 0.0:
        #    CustomDatePlot(self.hrc_times, self.hrc_table["2shldart"], plot=dp, label="HRC Shield Rate")
        #    h_all = np.concatenate([h_all, self.hrc_table["2shldart"]])
        dp.set_yscale("log")
        dp.set_ylabel("HRC Proxy (counts)")
        dp.set_ylim(0.5*np.nanmin(h_all), max(1.5*np.nanmax(h_all), 300))
        dp.add_hline(235.0, ls='--', lw=2, color="C3")
        self._plot_rzs(dp.ax)
        self._plot_comms(dp.ax)
        dp.set_legend(loc='upper left', fontsize=14)
        return dp

    def _plot_txings(self, plot=None):
        self.rates = defaultdict(list)
        first = True
        for i, o in enumerate(self.obsids):
            if o.obsid > 39999 and o.obsid < 60000:
                continue
            this_o = self.txings_data["obsid"] == o.obsid
            #print(i, o.obsid, this_o.sum())
            if this_o.sum() == 0:
                continue
            t = self.txings_data[this_o]
            this_time = self.txings_times[this_o].secs
            for k in ["fi_rate", "bi_rate", "fi_rate_limit", "bi_rate_limit"]:
                if not first:
                    self.rates[k].append([np.nan])
                self.rates[k].append(t[k])
            if not first:
                self.rates["times"].append([0.5*(this_time[0]+self.rates["times"][-1][-1])])
            self.rates["times"].append(this_time)
            first = False
        for k in self.rates:
            self.rates[k] = np.concatenate(self.rates[k])
        #print(self.rates["fi_rate"])
        plot = CustomDatePlot(self.rates["times"], self.rates["fi_rate"]*0.01, 
                              fmt='.', label="FI Rate", color="C0", lw=0, plot=plot)
        CustomDatePlot(self.rates["times"], self.rates["bi_rate"]*0.01, 
                       fmt='.', label="BI Rate", color="C1", lw=0, plot=plot)
        CustomDatePlot(self.rates["times"], self.rates["fi_rate_limit"]*0.01,
                       label="FI Rate", color="C0", plot=plot)
        CustomDatePlot(self.rates["times"], self.rates["bi_rate_limit"]*0.01,
                       label="BI Rate", color="C1", plot=plot)
        plot.set_yscale("log")
        plot.set_ylim(0.1, 100)
        plot.set_ylabel("ACIS Threshold Crossing Rate (cts/row/s)", fontsize=18)
        plot.ax.tick_params(which="major", width=2, length=6)
        plot.ax.tick_params(which="minor", width=2, length=3)
        fontProperties = font_manager.FontProperties(size=18)
        for label in plot.ax.get_xticklabels():
            label.set_fontproperties(fontProperties)
        for label in plot.ax.get_yticklabels():
            label.set_fontproperties(fontProperties)
        for axis in ['top', 'bottom', 'left', 'right']:
            plot.ax.spines[axis].set_linewidth(2)
        plot.set_legend(fontsize=18)
        self._plot_rzs(plot.ax)
        self._plot_comms(plot.ax)
        return plot

    def plot_txings(self):
        return self._plot_txings()

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

    def plot_goes_r(self):
        return self._plot_goes_r()

    def scatter_plots(self):
        fig, (ax1, ax2) = plt.subplots(figsize=(20, 9.5), ncols=2)
        hrc_shield = np.interp(self.ace_times.secs, self.hrc_times.secs, self.hrc_table["hrc_shield"])
        ax1.scatter(self.ace_table["p3"], hrc_shield)
        ax1.set_title("GOES HRC Proxy vs. ACE P3", fontsize=18)
        ax1.set_xlabel("ACE P3 Flux (particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)",
                      fontsize=18)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        fi_rate = np.empty(self.ace_times.secs.shape)
        fi_rate[:] = np.nan
        for o in self.obsids:
            if o.obsid > 39999:
                continue
            this_o = self.txings_data["obsid"] == o.obsid
            t = self.txings_data[this_o]
            idxs = np.searchsorted(self.ace_times.secs, t["time"])-1
            fi_rate[idxs] = t["fi_rate"]*0.01
        ax2.scatter(self.hrc_table["hrc_shield"], fi_rate)
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