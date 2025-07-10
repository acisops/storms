import time
from .utils import MLPModel, transform_goes, LogHyperbolicTangentScaler
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def find_near_detections(y_data, limits, tolerance=-10):
    # tolerance is negative so that anything near OR OVER the limit will be boosted
    return np.where(100 * (y_data - limits) / y_data >= tolerance)[0]


def test_split(X, y, index_11_fold):
    test_index = index_11_fold == 10
    X_test = X[test_index]
    y_test = y[test_index]
    return X_test, y_test, test_index


def train_val_split(X, y, index_11_fold, k):
    val_index = index_11_fold == k
    train_index = (index_11_fold != k) & (index_11_fold != 10)
    X_val = X[val_index]
    y_val = y[val_index]
    X_train = X[train_index]
    y_train = y[train_index]
    return X_train, y_train, X_val, y_val, val_index, train_index


rng = np.random.default_rng(12345)
# Verify MPS support
if torch.backends.mps.is_available():
    print("M1 GPU is available and PyTorch is configured to use MPS!")
else:
    print("MPS backend is not available. Check your setup.")
# Need to use "mps", cpu only allows for tiny models!
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Which k fold to start with
K_START = 0

# Which k fold to stop before
K_STOP = 10

x_factor = 10
DATA_INCREASE_FACTOR = 1


# There are FI and BI rates for txings, here I'm just choosing the FI rate
# "fi_rate", "bi_rate"
which_rate = "bi_rate"

# Need New Index fro k folds?
NEW_INDEX = True

# Load the FITS table which includes the txings data and the GOES proton data
t = Table.read(f"./{which_rate}_table.fits", format="fits")

# Figure out what columns from this table we need
add_cols = ["time", which_rate, "obsid", which_rate + "_limit"]
drop_chans = []

for col in t.colnames:
    # The data includes east and west columns, I am leaving the western data out
    # because it does not show up in real-time telemetry
    prefix = col.split("_")[0]
    print(prefix)
    if col.endswith("W") or prefix in drop_chans:
        continue
    if prefix.startswith("P"):
        add_cols.append(col)

print(add_cols)

# Create a new table
t_new = transform_goes(t[add_cols])
df = t_new.to_pandas()

# Get the labels and features.
X = df.drop([which_rate, which_rate + "_limit", "obsid", "time"], axis=1)
y = df[which_rate]

X_index = np.array(X.index, dtype=int)

if NEW_INDEX:
    index_11_fold = X_index % 11
    rng.shuffle(index_11_fold)
    np.save(
        "./Indices/" + which_rate + "_k_fold_validation_index.npy",
        index_11_fold,
    )
else:
    index_11_fold = np.load(
        "./Indices/" + which_rate + "_k_fold_validation_index.npy"
    )

X_data = np.array(X)
y_data = np.array(y)

INPUT_LENGTH = np.shape(X_data)[1]

# use this to modify dataset
near_detections_ids = find_near_detections(y_data, df[which_rate + "_limit"])


def normalization(X_data, y_data):
    X_means = np.mean(X_data, axis=0)
    Y_mean = np.mean(y_data)
    X_data_norm = np.empty(np.shape(X))

    for i in range(np.shape(X)[1]):
        X_data_norm[:, i] = np.tanh(np.log10((X_data[:, i] / X_means[i]) + 1))

    y_data_norm = np.tanh(np.log10((y_data / Y_mean) + 1))
    return X_data_norm, y_data_norm, X_means, Y_mean


# function that increases the number of ~limit detections and generally expands data set
# Use only for training and validation
def data_augmentation(
    X_data,
    y_data,
    data_index,
    near_detections_ids=near_detections_ids,
    epsilon=1e-4,
    X_INCREASE=DATA_INCREASE_FACTOR,
):
    # near_in_set = near or actual detections in data set
    # data_ids = location of intersection in data_index -> you need this to take data from train_set
    # id_ids = location of intersection in near_detections_ids - > don't need this
    near_in_set, data_ids, id_ids = np.intersect1d(
        data_index, near_detections_ids, return_indices=True
    )

    # number of additional near/actual detections needed to make data set balanced
    N_balancer = len(data_index) - (2 * len(near_in_set))
    N_aug_tot = N_balancer + len(data_index)
    print(
        len(near_in_set),
        N_balancer,
        N_aug_tot,
        (len(near_in_set) + N_balancer) / N_aug_tot,
    )

    # randomly select that many from available set
    aug_data_ids = rng.choice(data_ids, size=N_balancer)

    if X_INCREASE == 1:
        # initialize new arrays
        X_aug_data = np.empty((N_aug_tot, INPUT_LENGTH))
        y_aug_data = np.empty((N_aug_tot))

        # create new dataset index array
        data_index_aug = np.empty((N_aug_tot), dtype=int)
        data_index_aug[: len(data_index)] = data_index * 1
        data_index_aug[len(data_index) :] = data_index[aug_data_ids]

        # copy old data into new arrays
        X_aug_data[: len(data_index)] = 1.0 * X_data
        y_aug_data[: len(data_index)] = 1.0 * y_data

        # add new data into new arrays
        X_aug_data[len(data_index) :] = X_data[aug_data_ids] + (
            epsilon * rng.normal(size=(N_balancer, INPUT_LENGTH))
        )
        y_aug_data[len(data_index) :] = y_data[aug_data_ids] + (
            epsilon * rng.normal(size=N_balancer)
        )
    else:
        # initialize new arrays
        X_aug_data = np.empty((X_INCREASE * N_aug_tot, INPUT_LENGTH))
        y_aug_data = np.empty((X_INCREASE * N_aug_tot))

        X_aug_data_xth = np.empty((N_aug_tot, INPUT_LENGTH))
        y_aug_data_xth = np.empty((N_aug_tot))

        # create new dataset index array
        data_index_aug = np.empty((X_INCREASE * N_aug_tot), dtype=int)
        data_index_aug_xth = np.empty((N_aug_tot), dtype=int)
        data_index_aug_xth[: len(data_index)] = data_index * 1
        data_index_aug_xth[len(data_index) :] = data_index[aug_data_ids]

        # copy old data into new arrays
        X_aug_data_xth[: len(data_index)] = 1.0 * X_data
        y_aug_data_xth[: len(data_index)] = 1.0 * y_data

        # add new data into new arrays
        X_aug_data_xth[len(data_index) :] = X_data[aug_data_ids] + (
            epsilon * rng.normal(size=(N_balancer, INPUT_LENGTH))
        )
        y_aug_data_xth[len(data_index) :] = y_data[aug_data_ids] + (
            epsilon * rng.normal(size=N_balancer)
        )

        for i in range(X_INCREASE):
            data_index_aug[i * N_aug_tot : (i + 1) * N_aug_tot] = data_index_aug_xth
            X_aug_data[i * N_aug_tot : (i + 1) * N_aug_tot] = X_aug_data_xth + (
                epsilon * rng.normal(size=(N_aug_tot, INPUT_LENGTH))
            )
            y_aug_data[i * N_aug_tot : (i + 1) * N_aug_tot] = y_aug_data_xth + (
                epsilon * rng.normal(size=N_aug_tot)
            )

    return X_aug_data, y_aug_data, data_index_aug


def batch_retriever(index, max_batch_index, batch_size):
    if index == max_batch_index:
        return index * batch_size, -1
    else:
        return index * batch_size, (index * batch_size) + batch_size


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
    MODEL_NAME,
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
    print(MODEL_NAME, " training start")
    FIRST = True
    val_min = 1e4
    initial_epoch = 0
    last_min_epoch = 0

    epochs = 100000

    count = 0
    IN = False

    train_loss = []
    val_loss = []
    test_loss = []

    train_mae = []
    val_mae = []
    test_mae = []

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
            file_name = "./Models/" + MODEL_NAME + "_min_epoch"
            torch.save(model.state_dict(), file_name)
            last_min_epoch = k + 1
            count = 0
        elif count > max_count:
            print("Training Stopped by max_count trigger")
            print("Training Ended at Epoch " + str(k + 1))
            break
        else:
            count += 1
        if (time.time() - start) / 3600 > 50:
            print("Training Ended Early for Time")
            print("Training Ended at Epoch " + str(k + 1))
            break
    ###############################################
    ### PLOT AND SAVE EPOCHS
    ###############################################
    plt.figure(figsize=(30, 30))
    plt.title("Loss vs Epoch", fontsize=15)
    plt.plot(np.arange(len(train_loss)), train_loss, color="C0")
    plt.plot(np.arange(len(train_loss)), val_loss, color="C0", linestyle="dashed")
    plt.plot(np.arange(len(train_loss)), test_loss, color="C0", linestyle="dotted")
    plt.yscale("log")
    plt.legend(labels=["train_mse", "val_mse", "test_mse"], fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    #plt.show()
    plt.savefig("./Plots/" + MODEL_NAME + "epochs.png", bbox_inches="tight")
    plt.close()
    ###############################################
    ###############################################
    ###############################################


def k_fold_training(
    X_data,
    y_data,
    index_11_fold,
    which_rate,
    lr_factor=1e3,
    max_count=50,
    batch_size=32,
):
    lr = 1e-2 / lr_factor

    X_data_norm, y_data_norm, X_means, Y_mean = normalization(X_data, y_data)

    X_test, y_test, test_index = test_split(X_data_norm, y_data_norm, index_11_fold)
    test_length = len(X_test)

    for k in range(K_START, K_STOP):
        MODEL_NAME = (
            which_rate
            + "_k"
            + str(k)
            + "_"
            + str(x_factor)
            + "x_model_"
            + str(int(lr_factor))
            + "_learning_rate_max_stop_"
            + str(max_count)
            + "_DATA_AUG_"
            + str(DATA_INCREASE_FACTOR)
            + "x"
        )

        # Initialize the model
        model = MLPModel(INPUT_LENGTH, x_factor).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_train, y_train, X_val, y_val, val_index, train_index = train_val_split(
            X_data_norm, y_data_norm, index_11_fold, k
        )

        X_train, y_train, train_index = data_augmentation(X_train, y_train, train_index)
        X_val, y_val, val_index = data_augmentation(X_val, y_val, val_index)

        train_length = len(X_train)
        val_length = len(X_val)

        train_batch_indices = np.arange(train_length // batch_size, dtype=int)
        train_indices = np.arange(train_length, dtype=int)
        train_max_batch_index = train_batch_indices[-1]

        val_batch_indices = np.arange(val_length // batch_size, dtype=int)
        val_indices = np.arange(val_length, dtype=int)
        val_max_batch_index = val_batch_indices[-1]

        test_batch_indices = np.arange(test_length // batch_size, dtype=int)
        test_indices = np.arange(test_length, dtype=int)
        test_max_batch_index = test_batch_indices[-1]

        # train model
        training_loop(
            MODEL_NAME,
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

        model.load_state_dict(torch.load("./Models/" + MODEL_NAME + "_min_epoch"))

        # begin analysis of trained model
        y_train_pred = np.empty(np.shape(y_train))
        y_val_pred = np.empty(np.shape(y_val))
        y_test_pred = np.empty(np.shape(y_test))

        with torch.no_grad():
            model.eval()
            train_indices = np.arange(train_length)
            for i in train_batch_indices:
                j0, jf = batch_retriever(
                    train_batch_indices[i], train_max_batch_index, batch_size
                )
                x_batch = X_train[train_indices[j0:jf]]
                x_batch = torch.from_numpy(x_batch).to(device, torch.float32)
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
                x_batch = X_val[val_indices[j0:jf]]
                x_batch = torch.from_numpy(x_batch).to(device, torch.float32)
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
                x_batch = X_test[test_indices[j0:jf]]
                y_batch = y_test[test_indices[j0:jf]]
                x_batch = torch.from_numpy(x_batch).to(device, torch.float32)
                # ===================forward=====================
                y_test_pred[test_indices[j0:jf]] = (
                    model(x_batch).squeeze().cpu().detach().numpy()
                )
                del x_batch

            # Inverse transform the predictions and actual values
            y_train_pred_inv = Y_mean * (
                (10 ** (np.arctanh(y_train_pred))) - 1
            )  # scaler_y.inverse_transform(y_train_pred)
            y_val_pred_inv = Y_mean * (
                (10 ** (np.arctanh(y_val_pred))) - 1
            )  # scaler_y.inverse_transform(y_train_pred)
            y_test_pred_inv = Y_mean * (
                (10 ** (np.arctanh(y_test_pred))) - 1
            )  # scaler_y.inverse_transform(y_test_pred)

            y_train_inv = Y_mean * ((10 ** (np.arctanh(y_train))) - 1)
            y_val_inv = Y_mean * ((10 ** (np.arctanh(y_val))) - 1)
            y_test_inv = Y_mean * ((10 ** (np.arctanh(y_test))) - 1)

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1)
            plt.title("Train Set")
            plt.plot(y_train_inv, y_train_pred_inv, ".")
            plt.plot(
                np.linspace(0, np.max(y_train_inv)),
                np.linspace(0, np.max(y_train_inv)),
                color="k",
                linestyle="dashed",
            )
            plt.xlabel("True y")
            plt.ylabel("Pred y")
            plt.subplot(1, 3, 2)
            plt.title("Val Set")
            plt.plot(y_val_inv, y_val_pred_inv, ".")
            plt.plot(
                np.linspace(0, np.max(y_val_inv)),
                np.linspace(0, np.max(y_val_inv)),
                color="k",
                linestyle="dashed",
            )
            plt.xlabel("True y")
            plt.ylabel("Pred y")
            plt.subplot(1, 3, 3)
            plt.title("Test Set")
            plt.plot(y_test_inv, y_test_pred_inv, ".")
            plt.plot(
                np.linspace(0, np.max(y_test_inv)),
                np.linspace(0, np.max(y_test_inv)),
                color="k",
                linestyle="dashed",
            )
            plt.xlabel("True y")
            plt.ylabel("Pred y")
            plt.savefig(
                "./Plots/Model_Predictions_vs_Truth_" + MODEL_NAME + ".png",
                bbox_inches="tight",
            )
            plt.close()

            time_test = np.array(df.loc[test_index, "time"])
            time_val = np.array(df.loc[val_index, "time"])
            time_train = np.array(df.loc[train_index, "time"])

            limit_test = np.array(df.loc[test_index, which_rate + "_limit"])
            limit_val = np.array(df.loc[val_index, which_rate + "_limit"])
            limit_train = np.array(df.loc[train_index, which_rate + "_limit"])

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1)
            plt.title(k)
            plt.title("Train Set")
            plt.plot(time_train, y_train_inv, ".", alpha=0.5, label="True")
            plt.plot(time_train, y_train_pred_inv, ".", alpha=0.5, label="Pred")
            plt.xlabel("Time")
            plt.ylabel("Y")
            plt.legend()
            plt.subplot(1, 3, 2)
            plt.title("Val Set")
            plt.plot(time_val, y_val_inv, ".", alpha=0.5, label="True")
            plt.plot(time_val, y_val_pred_inv, ".", alpha=0.5, label="Pred")
            plt.xlabel("Time")
            plt.ylabel("Y")
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.title("Test Set")
            plt.plot(time_test, y_test_inv, ".", alpha=0.5, label="True")
            plt.plot(time_test, y_test_pred_inv, ".", alpha=0.5, label="Pred")
            plt.xlabel("Time")
            plt.ylabel("Y")
            plt.legend()
            plt.savefig("./Plots/" + MODEL_NAME + ".png", bbox_inches="tight")
            #plt.show()
            plt.close()

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


k_fold_training(X_data, y_data, index_11_fold, which_rate)
