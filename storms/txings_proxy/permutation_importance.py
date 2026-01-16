import numpy as np
import torch
from astropy.table import Table
from storms.txings_proxy.utils import prep_data, get_model, n_folds, models_path, use_cols
import joblib
from cxotime import CxoTime
import astropy.units as u
from IPython import embed
import matplotlib.pyplot as plt


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


t0 = CxoTime("2025:152:05:51:38.000")
which_rate = "fi_rate"

scaler_x = joblib.load(models_path / f"scaler_{which_rate}_x.pkl")
scaler_y = joblib.load(models_path / f"scaler_{which_rate}_y.pkl")

tstart = t0 - 2.0*u.day
tstop = t0 + 2.0*u.day

t = Table.read(f"{which_rate}_table.fits", format="fits")

X = torch.from_numpy(scaler_x.transform(prep_data(t))).to(device, torch.float32)
y = torch.from_numpy(scaler_y.transform(t[which_rate].data)).to(device, torch.float32)

importances = []

loss_fn = torch.nn.MSELoss()

for k in range(n_folds):
    model = get_model(which_rate, k)
    model.eval()
    with torch.no_grad():
        baseline_preds = model(X).squeeze().cpu().detach().numpy()
        baseline_loss = loss_fn(baseline_preds.squeeze(), y)

    imps = []
    for i in range(X.shape[1]):
        X_permuted = X.clone()
        X_permuted[:, i] = X_permuted[:, i][torch.randperm(X_permuted.size(0))]

        with torch.no_grad():
            permuted_preds = model(X_permuted)
            permuted_loss = loss_fn(permuted_preds.squeeze(), y)

        imps.append(permuted_loss - baseline_loss)

    importances.append(np.array(imps))

importances = np.array(importances).sum(axis=0)

# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [use_cols[i] for i in sorted_idx]
sorted_importances = np.array(importances)[sorted_idx]

norm_importances = sorted_importances / np.sum(np.abs(sorted_importances))

fig, ax = plt.subplots(figsize=(10, 15))
ax.barh(sorted_features, sorted_importances)
ax.set_xlabel("Increase in validation loss after permutation")
ax.set_title("Permutation Feature Importance")
ax.invert_yaxis()  # Most important on top
fig.tight_layout()
fig.savefig(f"{which_rate}_pm.png", dpi=300)

