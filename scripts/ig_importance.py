import numpy as np
import torch
import torch.nn as nn
from astropy.table import Table
from storms.txings_proxy.utils import prep_data, get_model, n_folds, models_path, use_cols, txings_path
import joblib
from cxotime import CxoTime
import astropy.units as u
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PartialInputWrapper(nn.Module):
    def __init__(self, model, keep_indices, total_features):
        super().__init__()
        self.model = model
        self.keep_indices = keep_indices
        self.total_features = total_features

    def forward(self, x_subset):
        # 1. Create a full-sized tensor of zeros on the same device as the input
        full_input = torch.zeros((x_subset.shape[0], self.total_features), device=x_subset.device)

        # 2. Map the subset back to the original feature positions
        full_input[:, self.keep_indices] = x_subset

        # 3. Pass the reconstructed input through the original model
        return self.model(full_input)


t0 = CxoTime("2025:152:05:51:38.000")
which_rate = "fi_rate"

scaler_x = joblib.load(models_path / f"scaler_{which_rate}_x.pkl")
scaler_y = joblib.load(models_path / f"scaler_{which_rate}_y.pkl")

tstart = t0 - 2.0*u.day
tstop = t0 + 2.0*u.day

t = Table.read(txings_path / f"{which_rate}_table.fits", format="fits")

tstart = CxoTime(tstart).secs
tstop = CxoTime(tstop).secs
idxs = (t["time"] >= tstart) & (t["time"] <= tstop)
t = t[idxs]

X = torch.from_numpy(scaler_x.transform(prep_data(t))).to(device, torch.float32)
y = torch.from_numpy(scaler_y.transform(t[which_rate].data)).to(device, torch.float32)

use_all = True

if use_all:
    keep_indices = list(range(len(use_cols)))
    use_str = ""
else:
    keep_indices = [i for i, col in enumerate(use_cols) if "ephem" not in col]
    use_str = "no_ephem"

filtered_cols = [use_cols[i] for i in keep_indices]

X_filtered = X[:, keep_indices]

scores = []

for k in range(n_folds):
    model = get_model(which_rate, k)
    model.eval()
    # Instantiate the wrapper
    wrapped_model = PartialInputWrapper(model, keep_indices, len(use_cols))
    ig = IntegratedGradients(wrapped_model)
    dataset = TensorDataset(X_filtered, y)
    loader = DataLoader(dataset, batch_size=16)
    total_attributions = []
    for batch_features, _ in loader:
        # We only need the features (batch_features) for attribution
        # target=0 refers to the single output node of your regression model
        batch_attr = ig.attribute(batch_features, target=0)
        total_attributions.append(batch_attr.detach())
    # Combine results
    all_attributions = torch.cat(total_attributions, dim=0)
    # Calculate global importance (average absolute attribution per feature)
    feature_importance = all_attributions.abs().mean(dim=0).cpu().numpy()
    scores.append(feature_importance)

scores = np.array(scores).mean(axis=0)

print(scores.min(), scores.max())

# Sort features by importance
sorted_idx = np.argsort(scores)[::-1]
sorted_features = [filtered_cols[i] for i in sorted_idx]
sorted_scores = np.array(scores)[sorted_idx]

fig, ax = plt.subplots(figsize=(10, 15))
ax.barh(sorted_features, sorted_scores)
ax.set_xlabel("Importance Score (Absolute Attribution)")
ax.set_title("Feature Importance using Integrated Gradients")
ax.invert_yaxis()  # Most important on top
fig.tight_layout()
fig.savefig(f"{which_rate}_ig.png", dpi=300)

#embed()
