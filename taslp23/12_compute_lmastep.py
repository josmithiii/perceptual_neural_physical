"""
This script calculates the M matrices and sigmas corresponding to each parameter
"""
import datetime
import functorch
import functools
import taslp23
import kymatio
import numpy as np
from pnp_synth.physical import ftm
import os
import pandas as pd
import pnp_synth
import h5py
import sklearn
import sys
import time
import torch

# Set device intelligently
if torch.backends.mps.is_available():
    device = torch.device("mps")
    precision = torch.float32  # MPS only supports float32
    print("Using MPS device with float32 precision")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    precision = torch.float32  # Keep consistent with original behavior
    print("Using CUDA device with float32 precision")
else:
    device = torch.device("cpu")
    precision = torch.float64  # CPU can handle float64
    print("Using CPU device with float64 precision")

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
id_start = int(sys.argv[2])
id_end = int(sys.argv[3])
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

dir_name = "M_log" # M, M_nominmax_log, M_log
synth_type = "ftm"
THETA_COLUMNS = ["omega", "tau", "p", "D", "alpha"]
eps = 1e-3

if "nominmax" in dir_name:
    scaler = None
    if "log" in dir_name:
        logscale = True
    else:
        logscale = False

    full_df = taslp23.load_fold(synth_type)
    nus = []
    for column in THETA_COLUMNS:
        if not logscale and column in ["omega", "p", "D"]:
            nus.append(10 ** full_df[column].values)
        else:
            nus.append(full_df[column].values)

    nus = np.stack(nus, axis=1)

else:
    if "log" in dir_name:
        logscale = True
    else:
        logscale = False

    _, scaler = taslp23.scale_theta(logscale, synth_type) #sorted in terms of id
    full_df = taslp23.load_fold(synth_type)
    nus = []
    for column in THETA_COLUMNS:
        if not logscale and column in ["omega", "p", "D"]:
            nus.append(10 ** full_df[column].values)
        else:
            nus.append(full_df[column].values)
    nus = np.stack(nus, axis=1)

# Define the forward PNP operator.
S_from_nu = taslp23.pnp_forward_factory(scaler, logscale, synth_type)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
dS_over_dnu = functorch.jacfwd(S_from_nu)

os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)
torch.autograd.set_detect_anomaly(True)
# Make h5 files for M
for fold in taslp23.FOLDS:
    fold_df = taslp23.load_fold(synth_type, fold)
    h5_name = "ftm_{}_M_{}.h5".format(fold, id_start)
    h5_path = os.path.join(save_dir, dir_name, h5_name)
    if not os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as h5_file:
            M_group = h5_file.create_group("M")
            evals_group = h5_file.create_group("sigma")

fold_df = taslp23.load_fold(synth_type)
for i in range(id_start, id_end):
    row = fold_df.iloc[i]
    #theta = torch.tensor([row[column] for column in setups.THETA_COLUMNS])
    key = int(row["ID"]) # index in the full dataframe
    fold = row['fold']
    h5_name = "ftm_{}_M_{}.h5".format(fold, id_start)
    h5_path = os.path.join(save_dir, dir_name, h5_name)
    nu = torch.tensor(nus[key, :], requires_grad=True, dtype=precision).to(device)

    with h5py.File(h5_path, "r") as h5_file:
        if str(i) not in h5_file['sigma'].keys():
            # Compute Jacobian: d(S) / d(nu)
            ismake = True
        else:
            ismake = False

    if ismake:
        J = dS_over_dnu(nu).detach()
        M = torch.matmul(J.T, J)
        assert M.shape[0] == 5 and M.shape[1] == 5

        # Append to HDF5 file
        with h5py.File(h5_path, "a") as h5_file:
            h5_file['M'][str(i)] = M.cpu()
            h5_file['sigma'][str(i)] = torch.linalg.eigvals(M).cpu()

# Print
now = str(datetime.datetime.now())
sys.stdout.flush()

# Empty line between folds
print("")


# Print elapsed time.
print(str(datetime.datetime.now()) + " Success.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.0
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
    elapsed_hours, elapsed_minutes, elapsed_seconds
)
print("Total elapsed time: " + elapsed_str + ".")
