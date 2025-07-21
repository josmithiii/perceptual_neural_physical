"""
This script calculates the JTFS coefficients and the associated Jacobian
with respect to each normalized parameter for FTM drum synthesis.

ICASSP25 version - computes PNP Jacobian matrices for ftm.rectangular_drum
"""
import datetime
import functools
import icassp25
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

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Set up parameters - ICASSP25 uses logscale and MinMax scaling
logscale = True
minmax = True

# Rescale shape parameters ("theta") to the interval [-1, 1] using MinMax scaler
nus, scaler = icassp25.scale_theta(logscale=logscale)

# Define the forward PNP operator
S_from_nu = icassp25.pnp_forward_factory(scaler, logscale=logscale)

# Define the associated Jacobian operator
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
dS_over_dnu = torch.func.jacfwd(S_from_nu)

# Create output directory for M matrices
dir_name = "ftm_M_log"
os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)

# Set anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Make h5 files for M matrices
for fold in icassp25.FOLDS:
    fold_df = icassp25.load_fold(fold)
    h5_name = "ftm_{}_M.h5".format(fold)
    h5_path = os.path.join(save_dir, dir_name, h5_name)

    # Create HDF5 file with groups for M matrices and eigenvalues
    with h5py.File(h5_path, "w") as h5_file:
        M_group = h5_file.create_group("M")
        evals_group = h5_file.create_group("sigma")

    print(f"Computing Jacobian matrices for {fold} fold ({len(fold_df)} samples)...")

    # Define row iterator
    row_iter = fold_df.iterrows()

    for i, row in row_iter:
        key = int(row["ID"])  # index in the full dataframe

        # Get the normalized parameters for this sample
        # Use CPU and appropriate dtype for JTFS-GPU compatibility
        device = torch.device("cpu")
        nu = torch.tensor(nus[key, :], requires_grad=True, dtype=torch.float32)
        nu = nu.to(device)

        try:
            # Compute Jacobian: d(S) / d(nu)
            # This computes the gradient of the JTFS representation S with respect
            # to the normalized parameters nu
            J = dS_over_dnu(nu).detach()

            # Compute the Gram matrix M = J^T * J
            # This gives us the PNP metric tensor used in the loss function
            M = torch.matmul(J.T, J)

            # Verify dimensions - ICASSP25 has 5 parameters
            assert M.shape[0] == 5 and M.shape[1] == 5, f"Expected 5x5 matrix, got {M.shape}"

            # Compute eigenvalues for analysis
            eigenvals = torch.linalg.eigvals(M)

            # Store in HDF5 file
            with h5py.File(h5_path, "a") as h5_file:
                h5_file['M'][str(i)] = M.cpu().numpy()
                h5_file['sigma'][str(i)] = eigenvals.cpu().numpy()

        except Exception as e:
            print(f"Error computing Jacobian for sample {i} (ID {key}): {e}")
            # Store zeros as fallback
            with h5py.File(h5_path, "a") as h5_file:
                h5_file['M'][str(i)] = np.zeros((5, 5))
                h5_file['sigma'][str(i)] = np.zeros(5)
            continue

        # Print progress periodically
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(fold_df)} samples...")
            sys.stdout.flush()

    # Print completion for this fold
    now = str(datetime.datetime.now())
    print(f"{now} Completed {fold} fold: {h5_path}")
    sys.stdout.flush()

    # Empty line between folds
    print("")

# Print elapsed time
print(str(datetime.datetime.now()) + " Success.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.0
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
    elapsed_hours, elapsed_minutes, elapsed_seconds
)
print("Total elapsed time: " + elapsed_str + ".")
