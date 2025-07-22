"""
FAST VERSION using MPS acceleration and batch processing.
This script calculates the JTFS coefficients and the associated Jacobian
with respect to each normalized parameter.
"""
import datetime
try:
    from torch.func import jacfwd, vmap
except ImportError:
    from functorch import jacfwd, vmap
import functools
import taslp23
import kymatio
import numpy as np
import os
import pandas as pd
import pnp_synth
import sklearn
import sys
import time
import torch

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs/taslp23"
id_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
id_end = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
logscale = int(sys.argv[4]) if len(sys.argv) > 4 else 1
synth_type_int = int(sys.argv[5]) if len(sys.argv) > 5 else 0
synth_type = "ftm" if synth_type_int == 0 else "amchirp"
minmax = int(sys.argv[6]) if len(sys.argv) > 6 else 1
force_cpu = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # Optional: force CPU for float64

print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [kymatio, np, pd, sklearn, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")

# Set device - use CPU for float64 precision, MPS for speed with float32
if force_cpu:
    device = torch.device("cpu")
    precision = torch.float64
    print(f"Using device: {device} with {precision} precision (forced CPU for precision)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    precision = torch.float32
    print(f"Using device: {device} with {precision} precision")
    print("Note: MPS uses float32. Use force_cpu=1 for float64 precision if needed.")
else:
    device = torch.device("cpu")
    precision = torch.float64
    print(f"Using device: {device} with {precision} precision")
sys.stdout.flush()

# Create folders
for fold in taslp23.FOLDS:
    os.makedirs(os.path.join(save_dir, "S", fold), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "J", fold), exist_ok=True)

# Load DataFrame
full_df = taslp23.load_fold(synth_type, "full")
params = full_df.values
n_samples = params.shape[0]
assert n_samples > id_end > id_start >= 0  # id is between 0 and (100k-1)

# Rescale shape parameters ("theta") to the interval [-1, 1].
if minmax:
    nus, scaler = taslp23.scale_theta(logscale, synth_type)
else:
    scaler = None

# Define the forward PNP operator.
S_from_nu = taslp23.pnp_forward_factory(scaler, logscale, synth_type)

# Define the associated Jacobian operator.
# NB: jacfwd is faster than reverse-mode autodiff here because the input
# is low-dimensional (5) whereas the output is high-dimensional (~1e4)
dS_over_dnu = jacfwd(S_from_nu)

# Batch processing configuration
BATCH_SIZE = 100  # Increased from 50 to utilize more GPU capacity

def process_batch_ids(batch_indices, nus_batch, full_df_batch):
    """Process a batch of samples with MPS acceleration."""
    device_nus = []
    file_paths = []

    for idx, nu_sample in enumerate(nus_batch):
        i = batch_indices[idx]
        fold = full_df_batch["fold"].iloc[idx]
        i_prefix = synth_type + "_" + str(i).zfill(len(str(n_samples)))
        S_path = os.path.join(save_dir, "S", fold, i_prefix + "_jtfs.npy")
        J_path = os.path.join(save_dir, "J", fold, i_prefix + "_grad_jtfs.npy")

        # Only process if files don't exist
        if not (os.path.exists(S_path) and os.path.exists(J_path)):
            nu_tensor = torch.tensor(nu_sample, requires_grad=True, device=device, dtype=precision)
            device_nus.append(nu_tensor)
            file_paths.append((S_path, J_path, fold, i_prefix))

    if not device_nus:
        return  # All files already exist

    # Process on device
    results = []
    for nu_tensor in device_nus:
        try:
            # Compute forward transformation and Jacobian
            S = S_from_nu(nu_tensor)
            J = dS_over_dnu(nu_tensor)

            # Move to CPU for saving
            results.append((S.detach().cpu(), J.detach().cpu()))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print(f"GPU memory error: {e}")
                print(f"Consider reducing BATCH_SIZE from {BATCH_SIZE} to a smaller value (e.g., 25 or 50)")
                print("You can edit the script directly or restart with more GPU memory available.")
                raise  # Re-raise to stop execution
            else:
                print(f"Runtime error processing sample: {e}")
                results.append(None)
        except Exception as e:
            print(f"Unexpected error processing sample: {e}")
            results.append(None)

    # Save results
    for idx, result in enumerate(results):
        if result is not None:
            S_cpu, J_cpu = result
            S_path, J_path, fold, i_prefix = file_paths[idx]

            np.save(S_path, S_cpu.numpy())
            np.save(J_path, J_cpu.numpy())

            now = str(datetime.datetime.now())
            print(now + " Exported: {}/{}".format(fold, i_prefix))
            sys.stdout.flush()

# Process in batches
total_samples = id_end - id_start
n_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Processing {total_samples} samples in {n_batches} batches of size {BATCH_SIZE}")
if device.type == "mps":
    print(f"Note: Using batch size {BATCH_SIZE} with MPS. If you encounter memory errors, reduce BATCH_SIZE in the script.")

torch.autograd.set_detect_anomaly(True)
for batch_idx in range(n_batches):
    start_idx = id_start + batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, id_end)
    batch_indices = list(range(start_idx, end_idx))

    # Extract batch data
    nus_batch = nus[start_idx:end_idx]
    full_df_batch = full_df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Process batch
    process_batch_ids(batch_indices, nus_batch, full_df_batch)

    # Progress update
    if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
        progress = (batch_idx + 1) / n_batches * 100
        print(f"Progress: Batch {batch_idx + 1}/{n_batches} ({progress:.1f}%)")
        sys.stdout.flush()

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
