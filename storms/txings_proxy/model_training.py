import time
from storms.txings_proxy.utils import MLPModel, transform_goes, LogHyperbolicTangentScaler, MultiScaler
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython import embed
import joblib
from sklearn.metrics import mean_squared_error


# Verify MPS support
if torch.backends.mps.is_available():
    print("M1 GPU is available and PyTorch is configured to use MPS!")
else:
    print("MPS backend is not available. Check your setup.")
# Need to use "mps", cpu only allows for tiny models!
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

base_path = Path(__file__).parent


def permutation_importance(model, X_val, y_val, loss_fn):
    model.eval()
    with torch.no_grad():
        baseline_preds = model(X_val)
        baseline_loss = loss_fn(baseline_preds, y_val).item()

    importances = []
    for i in range(X_val.shape[1]):
        X_permuted = X_val.clone()
        X_permuted[:, i] = X_permuted[:, i][torch.randperm(X_permuted.size(0))]

        with torch.no_grad():
            permuted_preds = model(X_permuted)
            permuted_loss = loss_fn(permuted_preds, y_val).item()

        importances.append(permuted_loss - baseline_loss)

    return np.array(importances)

def find_near_detections(y_data, limits, tolerance=-10):
    # tolerance is negative so that anything near OR OVER the limit will be boosted
    return np.where(100 * (y_data - limits) / y_data >= tolerance)[0]


def test_split(X, y, index_11_fold):
    test_index = index_11_fold == 10
    X_test = X[test_index]
    y_test = y[test_index]
    return X_test, y_test, np.where(test_index)[0]


def train_val_split(X, y, index_11_fold, k):
    val_index = index_11_fold == k
    train_index = (index_11_fold != k) & (index_11_fold != 10)
    X_val = X[val_index]
    y_val = y[val_index]
    X_train = X[train_index]
    y_train = y[train_index]
    return X_train, y_train, X_val, y_val, np.where(val_index)[0], np.where(train_index)[0]


def batch_retriever(index, max_batch_index, batch_size):
    if index == max_batch_index:
        return index * batch_size, -1
    else:
        return index * batch_size, index * batch_size + batch_size


# function that increases the number of ~limit detections and generally 
# expands data set. Use only for training and validation.
def data_augmentation(
    X_data,
    y_data,
    data_index,
    near_detections_ids,
    epsilon=1e-4,
    x_increase=1,
):
    input_length = X_data.shape[1]

    # near_in_set = near or actual detections in data set
    # data_ids = location of intersection in data_index -> you need this to take data from train_set
    # id_ids = location of intersection in near_detections_ids - > don't need this
    near_in_set, data_ids, _ = np.intersect1d(
        data_index, near_detections_ids, return_indices=True
    )

    # number of additional near/actual detections needed to make data set balanced
    n_near_in_set = len(near_in_set)
    n_data_index = len(data_index)
    N_balancer = n_data_index - 2 * n_near_in_set
    N_aug_tot = N_balancer + n_data_index
    print(
        n_near_in_set,
        N_balancer,
        N_aug_tot,
        (n_near_in_set + N_balancer) / N_aug_tot,
    )

    # randomly select that many from available set
    aug_data_ids = rng.choice(data_ids, size=N_balancer)

    if x_increase == 1:
        # initialize new arrays
        X_aug_data = np.empty((N_aug_tot, input_length))
        y_aug_data = np.empty(N_aug_tot)

        # create new dataset index array
        data_index_aug = np.empty(N_aug_tot, dtype=int)
        data_index_aug[:n_data_index] = data_index
        data_index_aug[n_data_index:] = data_index[aug_data_ids]

        # copy old data into new arrays
        X_aug_data[:n_data_index] = X_data.copy()
        y_aug_data[:n_data_index] = y_data.copy()

        # add new data into new arrays
        X_aug_data[n_data_index:] = X_data[aug_data_ids] + (
            epsilon * rng.normal(size=(N_balancer, input_length))
        )
        y_aug_data[n_data_index:] = y_data[aug_data_ids] + (
            epsilon * rng.normal(size=N_balancer)
        )
    else:
        # initialize new arrays
        X_aug_data = np.empty((x_increase * N_aug_tot, input_length))
        y_aug_data = np.empty(x_increase * N_aug_tot)

        X_aug_data_xth = np.empty((N_aug_tot, input_length))
        y_aug_data_xth = np.empty(N_aug_tot)

        # create new dataset index array
        data_index_aug = np.empty(x_increase * N_aug_tot, dtype=int)
        data_index_aug_xth = np.empty(N_aug_tot, dtype=int)
        data_index_aug_xth[:n_data_index] = data_index
        data_index_aug_xth[n_data_index:] = data_index[aug_data_ids]

        # copy old data into new arrays
        X_aug_data_xth[:n_data_index] = X_data.copy()
        y_aug_data_xth[:n_data_index] = y_data.copy()

        # add new data into new arrays
        X_aug_data_xth[n_data_index:] = X_data[aug_data_ids] + (
            epsilon * rng.normal(size=(N_balancer, input_length))
        )
        y_aug_data_xth[n_data_index:] = y_data[aug_data_ids] + (
            epsilon * rng.normal(size=N_balancer)
        )

        for i in range(x_increase):
            data_index_aug[i * N_aug_tot : (i + 1) * N_aug_tot] = data_index_aug_xth
            X_aug_data[i * N_aug_tot : (i + 1) * N_aug_tot] = X_aug_data_xth + (
                epsilon * rng.normal(size=(N_aug_tot, input_length))
            )
            y_aug_data[i * N_aug_tot : (i + 1) * N_aug_tot] = y_aug_data_xth + (
                epsilon * rng.normal(size=N_aug_tot)
            )

    return X_aug_data, y_aug_data, data_index_aug


def limit_testing(set_index, true_y, pred_y, limits):
    n_index = len(set_index)
    positives = np.zeros(n_index)
    negatives = np.zeros(n_index)
    true_positive = np.zeros(n_index)
    true_negative = np.zeros(n_index)
    false_positive = np.zeros(n_index)
    false_negative = np.zeros(n_index)
    for i in range(n_index):
        if true_y[i] >= limits[i]:
            positives[i] = 1
            if pred_y[i] >= limits[i]:
                true_positive[i] = 1
            elif pred_y[i] < limits[i]:
                false_negative[i] = 1
        else:
            negatives[i] = 1
            if pred_y[i] < limits[i]:
                true_negative[i] = 1
            if pred_y[i] >= limits[i]:
                false_positive[i] = 1
    return (
        true_positive,
        true_negative,
        false_positive,
        false_negative,
        positives,
        negatives,
    )


def training_loop(
    model_name,
    model,
    device,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    train_batch_indices,
    train_max_batch_index,
    train_indices,
    val_batch_indices,
    val_max_batch_index,
    val_indices,
    test_batch_indices,
    test_max_batch_index,
    test_indices,
    batch_size,
    criterion,
    optimizer,
    train_length,
    val_length,
    test_length,
    max_count,
):

    start = time.time()
    print(model_name, " training start")
    val_min = 1e4
    last_min_epoch = 0
    epochs = 100000
    count = 0

    train_loss = []
    val_loss = []
    test_loss = []

    print()
    for k in range(epochs + 1):
        # Training
        start_loop = time.time()
        model.train()
        train_loss_i = 0
        rng.shuffle(train_indices)
        for i in train_batch_indices:
            j0, jf = batch_retriever(
                train_batch_indices[i], train_max_batch_index, batch_size
            )
            x_batch = X_train[train_indices[j0:jf]]
            y_batch = y_train[train_indices[j0:jf]]

            x_batch = torch.from_numpy(x_batch).to(device, torch.float32)
            # ===================forward=====================
            y_batch_pred = model(x_batch)
            del x_batch

            y_batch = torch.from_numpy(y_batch).unsqueeze(1).to(device, torch.float32)
            loss = criterion(y_batch_pred, y_batch)
            train_loss_i += loss.item()
            del y_batch_pred, y_batch
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation & Testing
        with torch.no_grad():
            model.eval()
            val_loss_i = 0
            test_loss_i = 0

            rng.shuffle(val_indices)
            for i in val_batch_indices:
                j0, jf = batch_retriever(
                    val_batch_indices[i], val_max_batch_index, batch_size
                )
                x_batch = X_val[val_indices[j0:jf]]
                y_batch = y_val[val_indices[j0:jf]]
                x_batch = torch.from_numpy(x_batch).to(device, torch.float32)
                # ===================forward=====================
                y_batch_pred = model(x_batch)
                del x_batch
                y_batch = (
                    torch.from_numpy(y_batch).unsqueeze(1).to(device, torch.float32)
                )

                loss = criterion(y_batch_pred, y_batch)

                val_loss_i += loss.item()
                del y_batch_pred, y_batch

            rng.shuffle(test_indices)
            for i in test_batch_indices:
                j0, jf = batch_retriever(
                    test_batch_indices[i], test_max_batch_index, batch_size
                )
                x_batch = X_test[test_indices[j0:jf]]
                y_batch = y_test[test_indices[j0:jf]]
                x_batch = torch.from_numpy(x_batch).to(device, torch.float32)
                # ===================forward=====================
                y_batch_pred = model(x_batch)
                del x_batch
                y_batch = (
                    torch.from_numpy(y_batch).unsqueeze(1).to(device, torch.float32)
                )

                loss = criterion(y_batch_pred, y_batch)

                test_loss_i += loss.item()
                del y_batch_pred, y_batch
        # =====================log=======================

        if k % 100 == 0:
            print(
                "Loop Time:",
                (time.time() - start_loop) / 60,
                "Epoch: ",
                k + 1,
                "Train Loss:",
                train_loss_i / train_length,
                "Val Loss:",
                val_loss_i / val_length,
                "Test Loss:",
                test_loss_i / test_length,
                "Current Min Val Loss:",
                val_min,
                "Last Min Epoch",
                last_min_epoch,
                "Count:",
                count,
            )
            print()

        train_loss += [train_loss_i / train_length]
        val_loss += [val_loss_i / val_length]
        test_loss += [test_loss_i / test_length]

        if k > 5:
            if np.isnan(val_loss[k]):
                print("Failed")
                break

        val_test = val_loss[k]
        if val_test < val_min:
            val_min = val_test
            file_name = base_path / f"Models/{model_name}_min_epoch"
            torch.save(model.state_dict(), file_name)
            last_min_epoch = k + 1
            count = 0
        elif count > max_count:
            print("Training Stopped by max_count trigger")
            print(f"Training Ended at Epoch {k + 1}")
            break
        else:
            count += 1
        if (time.time() - start) / 3600 > 50:
            print("Training Ended Early for Time")
            print(f"Training Ended at Epoch {k + 1}")
            break

    # Plot and save epochs
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.plot(np.arange(len(train_loss)), train_loss, color="C0")
    ax.plot(np.arange(len(train_loss)), val_loss, color="C0", linestyle="dashed")
    ax.plot(np.arange(len(train_loss)), test_loss, color="C0", linestyle="dotted")
    ax.set_yscale("log")
    ax.set_title("Loss vs Epoch", fontsize=15)
    ax.legend(labels=["train_mse", "val_mse", "test_mse"], fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xlabel("Epochs", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    fig.savefig(base_path / f"Plots/{model_name}_epochs.png", bbox_inches="tight")


def k_fold_train_test(
    df,
    which_rate,
    new_index=False, # Need new index for k-folds?
    k_start=0,
    k_stop=10,
    lr_factor=1e3,
    max_count=50,
    batch_size=32,
    x_factor=10,
    data_increase_factor=1,
    no_train=False,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng()

    # Get the labels and features.
    X = df.drop([which_rate, f"{which_rate}_limit", "obsid", "time"], axis=1)
    y = df[which_rate]
    y_limit = df[f"{which_rate}_limit"]

    X_index = np.array(X.index, dtype=int)

    if new_index:
        index_11_fold = X_index % 11
        rng.shuffle(index_11_fold)
        np.save(
            base_path / f"Indices/{which_rate}_k_fold_validation_index.npy",
            index_11_fold,
        )
    else:
        index_11_fold = np.load(
            base_path / f"Indices/{which_rate}_k_fold_validation_index.npy",
        )

    feature_names = list(X.columns)
    print(feature_names)
    X_data = np.array(X)
    y_data = np.array(y)

    near_detections_ids = find_near_detections(y_data, y_limit)

    input_length = X_data.shape[1]

    lr = 1e-2 / lr_factor

    if no_train:
        #means = np.load(f"means_{which_rate}.npz")
        #scaler_x = LogHyperbolicTangentScaler(mean=means["x"])
        scaler_x = joblib.load(f"scaler_{which_rate}_x.pkl")
        scaler_y = joblib.load(f"scaler_{which_rate}_y.pkl")
        X_data_norm = scaler_x.transform(X_data)
        y_data_norm = scaler_y.transform(y_data)
    else:
        #scaler_x = LogHyperbolicTangentScaler()
        scaler_x = MultiScaler(X.columns.tolist())
        scaler_y = LogHyperbolicTangentScaler()
        X_data_norm = scaler_x.fit_transform(X_data)
        y_data_norm = scaler_y.fit_transform(y_data)
        joblib.dump(scaler_x, f"scaler_{which_rate}_x.pkl")
        joblib.dump(scaler_y, f"scaler_{which_rate}_y.pkl")
        #np.savez(f"means_{which_rate}.npz", x=scaler_x._mean, y=scaler_y._mean)

    X_test, y_test, test_index = test_split(X_data_norm, y_data_norm, index_11_fold)
    test_length = len(X_test)

    for k in range(k_start, k_stop):
        model_name = f"{which_rate}_k{k}_{x_factor}x_model_{int(lr_factor)}_learning_rate_max_stop_{max_count}_DATA_AUG_{data_increase_factor}x"

        X_train, y_train, X_val, y_val, val_index, train_index = train_val_split(
            X_data_norm, y_data_norm, index_11_fold, k
        )

        X_train, y_train, train_index = data_augmentation(X_train, y_train, train_index, near_detections_ids, x_increase=data_increase_factor)
        X_val, y_val, val_index = data_augmentation(X_val, y_val, val_index, near_detections_ids, x_increase=data_increase_factor)

        train_length = len(X_train)
        val_length = len(X_val)

        importances = permutation_importance(model, X_val, y_val, mean_squared_error)
        
        train_batch_indices = np.arange(train_length // batch_size, dtype=int)
        train_indices = np.arange(train_length, dtype=int)
        train_max_batch_index = train_batch_indices[-1]

        val_batch_indices = np.arange(val_length // batch_size, dtype=int)
        val_indices = np.arange(val_length, dtype=int)
        val_max_batch_index = val_batch_indices[-1]

        test_batch_indices = np.arange(test_length // batch_size, dtype=int)
        test_indices = np.arange(test_length, dtype=int)
        test_max_batch_index = test_batch_indices[-1]

        model = MLPModel(input_length, x_factor).to(device)

        if not no_train:
            # Initialize the model
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            # Train the model
            training_loop(
                model_name,
                model,
                device,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                train_batch_indices,
                train_max_batch_index,
                train_indices,
                val_batch_indices,
                val_max_batch_index,
                val_indices,
                test_batch_indices,
                test_max_batch_index,
                test_indices,
                batch_size,
                criterion,
                optimizer,
                train_length,
                val_length,
                test_length,
                max_count,
            )

        """
        # Suppose your training features are in a PyTorch tensor
        # Shape: (n_samples, n_features)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

        n_background = 100
        idx = np.random.choice(len(X_train_tensor), size=n_background, replace=False)
        background_data_tensor = X_train_tensor[idx]
        """

        model.load_state_dict(torch.load(base_path / f"Models/{model_name}_min_epoch"))

        """
        model.eval()

        """

        # begin analysis of trained model
        y_train_pred = np.empty(y_train.shape)
        y_val_pred = np.empty(y_val.shape)
        y_test_pred = np.empty(y_test.shape)

        with torch.no_grad():
            train_indices = np.arange(train_length)
            for i in train_batch_indices:
                j0, jf = batch_retriever(
                    train_batch_indices[i], train_max_batch_index, batch_size
                )
                x_batch = torch.from_numpy(X_train[train_indices[j0:jf]]).to(device, torch.float32)
                # ===================forward=====================
                y_train_pred[train_indices[j0:jf]] = (
                    model(x_batch).squeeze().cpu().detach().numpy()
                )
                del x_batch

            val_indices = np.arange(val_length)
            for i in val_batch_indices:
                j0, jf = batch_retriever(
                    val_batch_indices[i], val_max_batch_index, batch_size
                )
                x_batch = torch.from_numpy(X_val[val_indices[j0:jf]]).to(device, torch.float32)
                # ===================forward=====================
                y_val_pred[val_indices[j0:jf]] = (
                    model(x_batch).squeeze().cpu().detach().numpy()
                )
                del x_batch

            test_indices = np.arange(test_length)
            for i in test_batch_indices:
                j0, jf = batch_retriever(
                    test_batch_indices[i], test_max_batch_index, batch_size
                )
                x_batch = torch.from_numpy(X_test[test_indices[j0:jf]]).to(device, torch.float32)
                # ===================forward=====================
                y_test_pred[test_indices[j0:jf]] = (
                    model(x_batch).squeeze().cpu().detach().numpy()
                )
                del x_batch

            # Inverse transform the predictions and actual values
            y_train_pred_inv = scaler_y.inverse_transform(y_train_pred)
            y_val_pred_inv = scaler_y.inverse_transform(y_val_pred)
            y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)

            y_train_inv = scaler_y.inverse_transform(y_train)
            y_val_inv = scaler_y.inverse_transform(y_val)
            y_test_inv = scaler_y.inverse_transform(y_test)

            fig, axes = plt.subplots(ncols=3, figsize=(20, 5))
            axes[0].plot(y_train_inv, y_train_pred_inv, ".")
            equal_line_train = np.linspace(0, np.max(y_train_inv))
            axes[0].plot(
                equal_line_train, equal_line_train,
                color="k",
                linestyle="dashed",
            )
            axes[0].set_title("Train Set")
            axes[0].set_xlabel("True y")
            axes[0].set_ylabel("Pred y")
            axes[1].plot(y_val_inv, y_val_pred_inv, ".")
            equal_line_val = np.linspace(0, np.max(y_val_inv))
            axes[1].plot(
                equal_line_val, equal_line_val,
                color="k",
                linestyle="dashed",
            )
            axes[1].set_title("Val Set")
            axes[1].set_xlabel("True y")
            axes[1].set_ylabel("Pred y")
            axes[2].plot(y_test_inv, y_test_pred_inv, ".")
            equal_line_test = np.linspace(0, np.max(y_test_inv))
            axes[2].plot(
                equal_line_test, equal_line_test,
                color="k",
                linestyle="dashed",
            )
            axes[2].set_title("Test Set")
            axes[2].set_xlabel("True y")
            axes[2].set_ylabel("Pred y")
            fig.savefig(
                base_path / f"Plots/Model_Predictions_vs_Truth_{model_name}.png",
                bbox_inches="tight",
            )

            time_test = np.array(df.loc[test_index, "time"])
            time_val = np.array(df.loc[val_index, "time"])
            time_train = np.array(df.loc[train_index, "time"])

            limit_test = np.array(df.loc[test_index, f"{which_rate}_limit"])
            limit_val = np.array(df.loc[val_index, f"{which_rate}_limit"])
            limit_train = np.array(df.loc[train_index, f"{which_rate}_limit"])

            # Create a 1-row, 3-column subplot figure
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Train Set", "Val Set", "Test Set"))

            # Plot Train Set
            fig.add_trace(
                go.Scatter(x=time_train, y=y_train_inv, mode='markers', name='True (Train)', opacity=0.5),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_train, y=y_train_pred_inv, mode='markers', name='Pred (Train)', opacity=0.5),
                row=1, col=1
            )

            # Plot Validation Set
            fig.add_trace(
                go.Scatter(x=time_val, y=y_val_inv, mode='markers', name='True (Val)', opacity=0.5),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=time_val, y=y_val_pred_inv, mode='markers', name='Pred (Val)', opacity=0.5),
                row=1, col=2
            )

            # Plot Test Set
            fig.add_trace(
                go.Scatter(x=time_test, y=y_test_inv, mode='markers', name='True (Test)', opacity=0.5),
                row=1, col=3
            )
            fig.add_trace(
                go.Scatter(x=time_test, y=y_test_pred_inv, mode='markers', name='Pred (Test)', opacity=0.5),
                row=1, col=3
            )

            # Set common axis titles
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_xaxes(title_text="Time", row=1, col=3)
            fig.update_yaxes(title_text="Y", row=1, col=1)
            fig.update_yaxes(title_text="Y", row=1, col=2)
            fig.update_yaxes(title_text="Y", row=1, col=3)

            # Layout and legend
            fig.update_layout(
                width=1200, height=400,
                title_text=f"{model_name} Predictions",
                showlegend=True
            )

            # To save as HTML
            fig.write_html(str(base_path / f"Plots/{model_name}.html"))

            (
                true_positive_train,
                true_negative_train,
                false_positive_train,
                false_negative_train,
                positives_train,
                negatives_train,
            ) = limit_testing(train_index, y_train_inv, y_train_pred_inv, limit_train)

            (
                true_positive_val,
                true_negative_val,
                false_positive_val,
                false_negative_val,
                positives_val,
                negatives_val,
            ) = limit_testing(val_index, y_val_inv, y_val_pred_inv, limit_val)

            (
                true_positive_test,
                true_negative_test,
                false_positive_test,
                false_negative_test,
                positives_test,
                negatives_test,
            ) = limit_testing(test_index, y_test_inv, y_test_pred_inv, limit_test)

            print("K", k)
            print("TEST SET")
            print(
                "True Positives",
                np.sum(true_positive_test),
                "Actual Positives",
                np.sum(positives_test),
                "% of true",
                round(100 * np.sum(true_positive_test) / np.sum(positives_test), 1),
            )
            print(
                "False Positives",
                np.sum(false_positive_test),
                "Actual Positives",
                np.sum(positives_test),
                "% of reported",
                round(
                    100
                    * np.sum(false_positive_test)
                    / np.sum(true_positive_test + false_positive_test),
                    1,
                ),
            )
            print(
                "True Negatives",
                np.sum(true_negative_test),
                "Actual Negatives",
                np.sum(negatives_test),
                "% of true",
                round(100 * np.sum(true_negative_test) / np.sum(negatives_test)),
            )
            print(
                "False Negatives",
                np.sum(false_negative_test),
                "Actual Negatives",
                np.sum(negatives_test),
                "% of reported",
                round(
                    100
                    * np.sum(false_negative_test)
                    / np.sum(true_negative_test + false_negative_test),
                    1,
                ),
            )

            print("VALIDATION SET")
            print(
                "True Positives",
                np.sum(true_positive_val),
                "Actual Positives",
                np.sum(positives_val),
                "% of true",
                round(100 * np.sum(true_positive_val) / np.sum(positives_val), 1),
            )
            print(
                "False Positives",
                np.sum(false_positive_val),
                "Actual Positives",
                np.sum(positives_val),
                "% of reported",
                round(
                    100
                    * np.sum(false_positive_val)
                    / np.sum(true_positive_val + false_positive_val),
                    1,
                ),
            )
            print(
                "True Negatives",
                np.sum(true_negative_val),
                "Actual Negatives",
                np.sum(negatives_val),
                "% of true",
                round(100 * np.sum(true_negative_val) / np.sum(negatives_val)),
            )
            print(
                "False Negatives",
                np.sum(false_negative_val),
                "Actual Negatives",
                np.sum(negatives_val),
                "% of reported",
                round(
                    100
                    * np.sum(false_negative_val)
                    / np.sum(true_negative_val + false_negative_val),
                    1,
                ),
            )

            print("TRAINING SET")
            print(
                "True Positives",
                np.sum(true_positive_train),
                "Actual Positives",
                np.sum(positives_train),
                "% of true",
                round(100 * np.sum(true_positive_train) / np.sum(positives_train), 1),
            )
            print(
                "False Positives",
                np.sum(false_positive_train),
                "Actual Positives",
                np.sum(positives_train),
                "% of reported",
                round(
                    100
                    * np.sum(false_positive_train)
                    / np.sum(true_positive_train + false_positive_train),
                    1,
                ),
            )
            print(
                "True Negatives",
                np.sum(true_negative_train),
                "Actual Negatives",
                np.sum(negatives_train),
                "% of true",
                round(100 * np.sum(true_negative_train) / np.sum(negatives_train)),
            )
            print(
                "False Negatives",
                np.sum(false_negative_train),
                "Actual Negatives",
                np.sum(negatives_train),
                "% of reported",
                round(
                    100
                    * np.sum(false_negative_train)
                    / np.sum(true_negative_train + false_negative_train),
                    1,
                ),
            )
            print()


rng = np.random.default_rng(12345)

# There are FI and BI rates for txings, here I'm just choosing the FI rate
# "fi_rate", "bi_rate"
which_rate = "fi_rate"

# Load the FITS table which includes the txings data and the GOES proton data
t = Table.read(base_path / f"{which_rate}_table.fits", format="fits")

# Figure out what columns from this table we need
add_cols = ["time", which_rate, "obsid", which_rate + "_limit"]

drop_chans = []

for col in t.colnames:
    # The data includes east and west columns, I am leaving the western data out
    # because it does not show up in real-time telemetry
    prefix = col.split("_")[0]
    if col.endswith("W") or prefix in drop_chans:
        continue
    if prefix.startswith("P"):
        add_cols.append(col)
    if "ephem" in col:
        add_cols.append(col)

print(add_cols)

# Create a new table
t_new = transform_goes(t[add_cols])
df = t_new.to_pandas()

k_fold_train_test(df, which_rate, new_index=True, no_train=False, rng=rng)

