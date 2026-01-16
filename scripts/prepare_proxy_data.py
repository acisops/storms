import numpy as np
from astropy.table import Table
from cheta import fetch_sci as fetch
from cxotime import CxoTime
from mica.archive.cda import get_ocat_local
from scipy.ndimage import uniform_filter1d

from storms import SolarWind
from storms.txings_proxy.utils import goes_path, txings_path

t = Table.read(goes_path / "goes_16_18.fits")

print(CxoTime(t["time"][0]).yday, CxoTime(t["time"][-1]).yday)

sw = SolarWind(
    CxoTime(t["time"][0]).yday,
    CxoTime(t["time"][-1]).yday,
    get_txings=True,
    get_ace=False,
    get_hrc=False,
    get_goes=False,
)

obsids = np.unique(sw.txings_data["obsid"])

bright_obsids = [23539, 28052]
for obsid in obsids:
    if 38000 < int(obsid) < 60000:
        continue
    obsid_info = get_ocat_local(obsid=obsid)
    if (
        obsid_info["grat"] in ["LETG", "HETG"]
        and obsid_info["instr"].startswith("ACIS")
        or obsid_info["simode"].startswith("CC_")
    ):
        bright_obsids.append(obsid)

for col in t.colnames:
    if col.startswith("P"):
        t[col][np.isnan(t[col])] = 0.0
        t[col] = uniform_filter1d(t[col], 10, axis=0, mode="nearest")

chandra_ephems = [f"{obj}ephem0_{ax}" for obj in ["orbit", "solar"] for ax in "xyz"]

msids = fetch.MSIDset(chandra_ephems, t["time"][0], t["time"][-1], stat="5min")
msids.interpolate(times=t["time"])

fi_rate_table = Table()
bi_rate_table = Table()
bad_fi = sw.txings_data["fi_rate"] <= 0
bad_fi |= np.isin(sw.txings_data["obsid"], bright_obsids)
fi_rate_table["time"] = sw.txings_times.secs[~bad_fi]
fi_rate_table["fi_rate"] = sw.txings_data["fi_rate"][~bad_fi]
fi_rate_table["fi_rate_limit"] = sw.txings_data["fi_rate_limit"][~bad_fi]
fi_rate_table["obsid"] = sw.txings_data["obsid"][~bad_fi]
bad_bi = sw.txings_data["bi_rate"] <= 0
bad_bi |= np.isin(sw.txings_data["obsid"], bright_obsids)
bi_rate_table["time"] = sw.txings_times.secs[~bad_bi]
bi_rate_table["bi_rate"] = sw.txings_data["bi_rate"][~bad_bi]
bi_rate_table["bi_rate_limit"] = sw.txings_data["bi_rate_limit"][~bad_bi]
bi_rate_table["obsid"] = sw.txings_data["obsid"][~bad_bi]
for col in t.colnames:
    if col.startswith("P"):
        fi_rate_table[f"{col}_E"] = np.zeros_like(fi_rate_table["time"])
        fi_rate_table[f"{col}_W"] = np.zeros_like(fi_rate_table["time"])
        bi_rate_table[f"{col}_E"] = np.zeros_like(bi_rate_table["time"])
        bi_rate_table[f"{col}_W"] = np.zeros_like(bi_rate_table["time"])
        for obsid in obsids:
            mask = fi_rate_table["obsid"] == obsid
            if mask.sum() == 0:
                continue
            for i, ax in zip([0, 1], ["E", "W"], strict=False):
                fi_rate_table[f"{col}_{ax}"][mask] = 10 ** np.interp(
                    fi_rate_table["time"][mask], t["time"], np.log10(t[col][:, i])
                )
        for obsid in obsids:
            mask = bi_rate_table["obsid"] == obsid
            if mask.sum() == 0:
                continue
            for i, ax in zip([0, 1], ["E", "W"], strict=False):
                bi_rate_table[f"{col}_{ax}"][mask] = 10 ** np.interp(
                    bi_rate_table["time"][mask], t["time"], np.log10(t[col][:, i])
                )

for col in chandra_ephems:
    fi_rate_table[col] = np.zeros_like(fi_rate_table["time"])
    bi_rate_table[col] = np.zeros_like(bi_rate_table["time"])
    for obsid in obsids:
        mask = fi_rate_table["obsid"] == obsid
        if mask.sum() == 0:
            continue
        fi_rate_table[col][mask] = np.interp(
            fi_rate_table["time"][mask], msids[col].times, msids[col].vals
        )
    for obsid in obsids:
        mask = bi_rate_table["obsid"] == obsid
        if mask.sum() == 0:
            continue
        bi_rate_table[col][mask] = np.interp(
            bi_rate_table["time"][mask], msids[col].times, msids[col].vals
        )

fi_rate_table.write(txings_path / "fi_rate_table.fits", format="fits", overwrite=True)
bi_rate_table.write(txings_path / "bi_rate_table.fits", format="fits", overwrite=True)
