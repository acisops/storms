import argparse

import astropy.units as u
from cxotime import CxoTime
from ruamel.yaml import YAML

from storms import SolarWind


def main():
    parser = argparse.ArgumentParser(
        description="Plot solar wind data for a given time range."
    )
    parser.add_argument(
        "infile", type=str, help="The YAML file containing the parameters for plotting."
    )
    args = parser.parse_args()

    yaml = YAML()
    with open(args.infile) as f:
        params = yaml.load(f)

    basename = params["basename"]

    scs_107 = CxoTime(params["scs_107"])
    rts = CxoTime(params["rts"])
    ecs_start = CxoTime(params["ecs_start"]) if "ecs_start" in params else None
    ecs_stop = CxoTime(params["ecs_stop"]) if "ecs_stop" in params else None

    xmin = CxoTime(params["start"])
    xmax = CxoTime(params["stop"])
    start = xmin - 1.0 * u.day
    stop = xmax + 1.0 * u.day
    xmin2 = scs_107 - 5.0 * u.hr
    xmax2 = scs_107 + 1.0 * u.hr

    legend_loc = params.get("legend_loc", "lower left")

    sw = SolarWind(
        start,
        stop,
        use_browse=True,
        get_txings=True,
        get_states=True,
        get_goes=True,
        get_hrc=True,
    )

    def plot_lines(dp):
        props = {
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.75,
            "linewidth": 0,
        }
        dp.add_vline(rts, color="g", zorder=100)
        dp.add_text(
            rts + 1.0 * u.hr,
            0.15,
            "RTS",
            color="g",
            rotation=90,
            bbox=props,
            transform=dp.ax.get_xaxis_transform(),
        )
        dp.add_vline(scs_107, color="r", zorder=100)
        dp.add_text(
            scs_107 + 1.0 * u.hr,
            0.12,
            "SCS 107",
            color="r",
            rotation=90,
            bbox=props,
            transform=dp.ax.get_xaxis_transform(),
        )
        if ecs_start is not None:
            dp.add_vline(ecs_start, zorder=100)
            dp.add_text(
                ecs_start + 1.0 * u.hr,
                0.7,
                "Long ECS Start",
                color="g",
                rotation=90,
                bbox=props,
                transform=dp.ax.get_xaxis_transform(),
            )
        if ecs_stop is not None:
            dp.add_vline(ecs_stop, zorder=100)
            dp.add_text(
                ecs_stop + 1.0 * u.hr,
                0.7,
                "Long ECS Stop",
                color="g",
                rotation=90,
                bbox=props,
                transform=dp.ax.get_xaxis_transform(),
            )

    def plot_obsids_txings(sw, dp, xmin, xmax, label_offset):
        tmin, tmax = dp.ax.get_xlim()
        for o in sw.obsids:
            tstart = max(CxoTime(xmin).secs, o.tstart)
            tstop = min(CxoTime(xmax).secs, o.tstop)
            ostart = CxoTime((label_offset * tstart + (1.0 - label_offset) * tstop))
            if o.obsid > 39999 or (ostart.plot_date < tmin or ostart.plot_date > tmax):
                continue
            dp.add_text(
                ostart,
                6.0,
                o.obsid,
                color="C3",
                horizontalalignment="left",
                fontsize=14,
                zorder=100,
                rotation=90,
            )

    print("Plotting ACE P3 Flux.")
    dp = sw.plot_ace_p3()
    if "ace_p3_limits" in params:
        dp.set_ylim(*params["ace_p3_limits"])
    dp.set_xlim(xmin, xmax)
    plot_lines(dp)
    dp.set_legend(loc=legend_loc, fontsize=15, ncols=3, zorder=200)
    dp.savefig(f"ace_p3_{basename}.png")

    print("Plotting ACE Proton Flux.")
    dp = sw.plot_ace_p()
    if "ace_p_limits" in params:
        dp.set_ylim(*params["ace_p_limits"])
    dp.set_xlim(xmin, xmax)
    plot_lines(dp)
    dp.set_legend(loc=legend_loc, fontsize=15, ncols=2, zorder=200)
    dp.savefig(f"ace_p_{basename}.png")

    print("Plotting txings rates.")
    dp = sw.plot_txings()
    if "txings_limits" in params:
        dp.set_ylim(*params["txings_limits"])
    plot_lines(dp)
    dp.set_xlim(xmin, xmax)
    plot_obsids_txings(sw, dp, xmin, xmax, 0.5)
    dp.savefig(f"txings_{basename}.png")

    print("Plotting zoom-in of txings rates.")
    dp = sw.plot_txings(ms=10)
    if "txings_limits" in params:
        dp.set_ylim(*params["txings_limits"])
    dp.set_xlim(xmin2, xmax2)
    plot_obsids_txings(sw, dp, xmin2, xmax2, 0.5)
    dp.add_vline(scs_107, color="r")
    dp.add_text(scs_107 + 0.1 * u.hr, 0.5, "SCS 107", color="r", rotation=90)
    dp.savefig(f"txings_zoomin_{basename}.png")

    print("Plotting HRC Proxy.")
    dp = sw.plot_hrc()
    dp.set_xlim(xmin, xmax)
    if "hrc_limits" in params:
        dp.set_ylim(*params["hrc_limits"])
    plot_lines(dp)
    dp.set_legend(loc=legend_loc, fontsize=15, ncols=2, zorder=200)
    dp.savefig(f"hrc_proxy_{basename}.png")

    print("Plotting Goes Proton Flux.")
    dp = sw.plot_goes_r()
    dp.set_xlim(xmin, xmax)
    plot_lines(dp)
    if "goes_limits" in params:
        dp.set_ylim(*params["goes_limits"])
    dp.set_legend(loc=legend_loc, fontsize=15, ncols=2, zorder=200)
    dp.savefig(f"goes_p_{basename}.png")

    print("Plotting ACE Electron Flux.")
    dp = sw.plot_ace_e()
    dp.set_xlim(xmin, xmax)
    if "ace_e_limits" in params:
        dp.set_ylim(*params["ace_e_limits"])
    plot_lines(dp)
    dp.set_legend(loc=legend_loc, fontsize=15, ncols=2, zorder=200)
    dp.savefig(f"ace_e_{basename}.png")


if __name__ == "__main__":
    main()
