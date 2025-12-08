import requests
import time
import sys
from collections import defaultdict
import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
from storms.txings_proxy.utils import coeffs_16_19

goes_url_base = "https://services.swpc.noaa.gov/json/goes/"
goes_url_6h = "differential-protons-6-hour.json"
goes_url_7d = "differential-protons-7-day.json"
BAD_VALUE = -1.0e5


def get_goes_json_data(url):
    """
    Open the json file and return it as an astropy table
    """
    print(url)
    last_err = None
    for _ in range(3):
        try:
            json_file = requests.get(url)
            data = json_file.json()
            break
        except Exception as err:
            last_err = err
            time.sleep(5)
    else:
        print(f"Warning: failed to open URL {url}: {last_err}")
        return

    return Table(data)


def format_goes_proton_data(dat, break_time):
    """
    Manipulate the data and return them in a desired format.
    """

    first_time = Time(dat[0]["time_tag"]).cxcsec-60
    # Create a dictionary to capture the channel data for each time
    out = defaultdict(dict)
    for row in dat:
        if Time(row["time_tag"]).cxcsec-60 <= break_time:
            continue
        satellite = 16 if row['satellite'] == 19 else row['satellite']
        out[row["time_tag"]][f"{row['channel'].upper()}_g{satellite}_E"] = row["flux"]
        out[row["time_tag"]]["yaw_flip"] = row["yaw_flip"]

    if len(out.keys()) == 0:
        return None, None

    # Reshape that data into a table with the channels as columns
    newdat = Table(list(out.values())).filled(BAD_VALUE)
    newdat["time_tag"] = list(out.keys())  # Already in time order if dat rows in order

    # Add some time columns
    times = Time(newdat["time_tag"])
    newdat["time"] = times.cxcsec-60
    newdat["mjd"] = times.mjd.astype(int)
    newdat["secs"] = np.array(
        np.round((times.mjd - newdat["mjd"]) * 86400, decimals=0)
    ).astype(int)
    newdat["year"] = [t.year for t in times.datetime]
    newdat["month"] = [t.month for t in times.datetime]
    newdat["dom"] = [t.day for t in times.datetime]
    newdat["hhmm"] = np.array(
        [f"{t.hour}{t.minute:02}" for t in times.datetime]
    ).astype(int)

    for col in newdat.colnames:
        if col.startswith("P") and "g16" in col:
            newdat[col] *= coeffs_16_19[col.split("_")[0]]
    return newdat, first_time


def get_realtime_goes(break_time, use7d=False):
    dat_all = Table()
    url_end = goes_url_7d if use7d else goes_url_6h
    for source in ["primary", "secondary"]:
        url = f"{goes_url_base}{source}/{url_end}"
        dat = get_goes_json_data(url=url)
        if dat is None:
            return
        dat_all = vstack([dat_all, dat])
    return format_goes_proton_data(dat_all, break_time)
