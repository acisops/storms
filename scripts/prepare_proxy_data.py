import numpy as np
from astropy.table import Table
from cxotime import CxoTime
from mica.archive.cda import get_ocat_local
from numpy.testing import assert_allclose
from scipy.ndimage import uniform_filter1d

from storms import SolarWind
from storms.txings_proxy.utils import goes_path, coeffs_16_19

t16 = Table.read(goes_path / "goes_16.fits")
t18 = Table.read(goes_path / "goes_18.fits")
t19 = Table.read(goes_path / "goes_19.fits")

t = Table()
for col in t16.colnames:
    a = t16[col][
        (t16["time"] >= t18["time"][0]) & (t16["time"] < CxoTime("2025:097").secs)
    ]
    b = t19[col][t19["time"] >= CxoTime("2025:097").secs]
    if col.startswith("P"):
        coeff = coeffs_16_19[col]
    else:
        coeff = 1.0
    t[f"{col}_g16"] = np.append(a, coeff*b, axis=0)
for col in t18.colnames:
    a = t18[col][t18["time"] >= t18["time"][0]]
    t[f"{col}_g18"] = a
all_bad = np.zeros(len(t), dtype=bool)
for col in t.colnames:
    if "time" not in col and "yaw" not in col:
        bad = t[col].mask.sum(axis=1).astype("bool")
        all_bad = all_bad | bad
t = t[~all_bad]
assert_allclose(t["time_g16"], t["time_g18"])
t["time"] = t["time_g16"].copy()
for col in t.colnames:
    print(np.isnan(t[col]).any())
t.remove_columns(["time_g16", "time_g18"])
t.write("goes_16_18.fits", overwrite=True)

sw = SolarWind(CxoTime(t["time"][0]).yday, CxoTime(t["time"][-1]).yday, get_txings=True)

obsids = np.unique(sw.txings_data["obsid"])

bright_obsids = [23539, 28052]
for obsid in obsids:
    if 38000 < int(obsid) < 60000:
        continue
    obsid_info = get_ocat_local(obsid=obsid)
    if obsid_info["grat"] in ["LETG", "HETG"] and obsid_info["instr"].startswith(
        "ACIS"
    ):
        bright_obsids.append(obsid)

for col in t.colnames:
    if col.startswith("P"):
        t[col][np.isnan(t[col])] = 0.0
        t[col] = uniform_filter1d(t[col], 10, axis=0, mode="nearest")

fi_rate_table = Table()
bad = sw.txings_data["fi_rate"] <= 0
bad |= np.isin(sw.txings_data["obsid"], bright_obsids)
fi_rate_table["time"] = sw.txings_times.secs[~bad]
fi_rate_table["fi_rate"] = sw.txings_data["fi_rate"][~bad]
fi_rate_table["fi_rate_limit"] = sw.txings_data["fi_rate_limit"][~bad]
fi_rate_table["obsid"] = sw.txings_data["obsid"][~bad]
for col in t.colnames:
    if col.startswith("P"):
        print(col)
        fi_rate_table[f"{col}_E"] = np.zeros_like(fi_rate_table["time"])
        fi_rate_table[f"{col}_W"] = np.zeros_like(fi_rate_table["time"])
        for obsid in obsids:
            mask = fi_rate_table["obsid"] == obsid
            if mask.sum() == 0:
                continue
            for i, ax in zip([0, 1], ["E", "W"], strict=False):
                fi_rate_table[f"{col}_{ax}"][mask] = 10 ** np.interp(
                    fi_rate_table["time"][mask], t["time"], np.log10(t[col][:, i])
                )
fi_rate_table.write("fi_rate_table.fits", format="fits", overwrite=True)

bi_rate_table = Table()
bad = sw.txings_data["bi_rate"] <= 0
bad |= np.isin(sw.txings_data["obsid"], bright_obsids)
bi_rate_table["time"] = sw.txings_times.secs[~bad]
bi_rate_table["bi_rate"] = sw.txings_data["bi_rate"][~bad]
bi_rate_table["bi_rate_limit"] = sw.txings_data["bi_rate_limit"][~bad]
bi_rate_table["obsid"] = sw.txings_data["obsid"][~bad]
for col in t.colnames:
    if col.startswith("P"):
        bi_rate_table[f"{col}_E"] = np.zeros_like(bi_rate_table["time"])
        bi_rate_table[f"{col}_W"] = np.zeros_like(bi_rate_table["time"])
        for obsid in obsids:
            mask = bi_rate_table["obsid"] == obsid
            if mask.sum() == 0:
                continue
        for i, ax in zip([0, 1], ["E", "W"], strict=False):
            bi_rate_table[f"{col}_{ax}"] = 10 ** np.interp(
                bi_rate_table["time"], t["time"], np.log10(t[col][:, i])
            )
bi_rate_table.write("bi_rate_table.fits", format="fits", overwrite=True)
