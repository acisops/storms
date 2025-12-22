import torch
import numpy as np

from storms.utils import base_path, ephem_msids, goes_bands, use_cols
from cheta import fetch_sci as fetch
from scipy.ndimage import uniform_filter1d


models_path = base_path / "txings_proxy/Models"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def prep_data(t_goes):


    ephem_data = fetch.MSIDset(ephem_msids, ephem_start, ephem_stop, stat="5min")

    for msid in ephem_msids:
        t_goes[msid] = np.interp(
            t_goes["time"], ephem_data[msid].times, ephem_data[msid].vals
        )

    df = t_goes[use_cols].to_pandas()

    # Convert the specific flux to total flux in the band for each channel by multiplying
    # by the channel width
    for col in df.columns:
        if col.startswith("P"):
            prefix = col.split("_")[0]
            i = use_cols.index(col)
            df[col] = uniform_filter1d(df[col], 10, axis=0) * (
                    goes_bands[prefix][1] - goes_bands[prefix][0]
            )

    X = np.array(df)


