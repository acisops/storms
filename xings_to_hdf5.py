from astropy.io import ascii
import h5py
from IPython import embed

names = [
    "processing_phase",
    "run_index",
    "obsid",
    "processing_mode",
    "num_samples",
    "times",
    "I0",
    "I1",
    "I2",
    "I3",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
]

t = ascii.read("xings.txt", names=names)

#embed()

with h5py.File("xings.h5", "w") as f:
    for name in names:
        if name.startswith("processing"):
            data = t[name].data.astype("S")
        else:
            data = t[name].data
        f.create_dataset(name, data=data)
