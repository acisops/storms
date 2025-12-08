import numpy as np
from astropy.table import Table
from cheta import fetch_sci as fetch
from cxotime import CxoTime
from mica.archive.cda import get_ocat_local
from numpy.testing import assert_allclose
from scipy.ndimage import uniform_filter1d
import numpy.ma as ma

from storms import SolarWind
from storms.txings_proxy.utils import goes_path, coeffs_16_19

t16 = Table.read(goes_path / "goes_16.fits")
t18 = Table.read(goes_path / "goes_18.fits")
t19 = Table.read(goes_path / "goes_19.fits")

print(CxoTime(t18["time"][0]).yday, CxoTime(t18["time"][-1]).yday)

t = Table()
for col in t16.colnames:
    print(col)
    a = t16[col][
        (t16["time"] >= t18["time"][0]) & (t16["time"] < CxoTime("2025:097").secs)
        ]
    b = t19[col][t19["time"] >= CxoTime("2025:097").secs]
    if col.startswith("P"):
        coeff = coeffs_16_19[col]
    else:
        coeff = 1.0
    t[f"{col}_g16"] = ma.append(a, coeff * b, axis=0)
for col in t18.colnames:
    a = t18[col][t18["time"] >= t18["time"][0]]
    t[f"{col}_g18"] = a
all_bad = np.zeros(len(t), dtype=bool)

for col in t.colnames:
    if "time" in col or "yaw" in col:
        continue
    if col.startswith("P"):
        bad = t[col].mask.sum(axis=1).astype("bool")
    elif "ephem" in col:
        bad = t[col].mask.astype("bool")
    #else:
    #    bad = np.zeros_like(t[col])
    all_bad = all_bad | bad
    print(col, all_bad.sum(), bad.sum(), t[col].shape)
print(len(t), all_bad.sum())
t = t[~all_bad]
assert_allclose(t["time_g16"], t["time_g18"])
t["time"] = t["time_g16"].copy()
# for col in t.colnames:
#    print(np.isnan(t[col]).any())
t.remove_columns(["time_g16", "time_g18"])
t.write("goes_16_18.fits", overwrite=True)

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
        fi_rate_table[col][mask] = np.interp(fi_rate_table["time"][mask], msids[col].times, msids[col].vals)
    for obsid in obsids:
        mask = bi_rate_table["obsid"] == obsid
        if mask.sum() == 0:
            continue
        bi_rate_table[col][mask] = np.interp(bi_rate_table["time"][mask], msids[col].times, msids[col].vals)

fi_rate_table.write("fi_rate_table.fits", format="fits", overwrite=True)
bi_rate_table.write("bi_rate_table.fits", format="fits", overwrite=True)
