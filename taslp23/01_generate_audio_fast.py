"""
FAST VERSION of 01_generate_audio.py using MPS acceleration and batching.
This script computes 100k drum sounds by solving a 4th-order partial
differential equation with the pnp_synth.ftm submodule.
FTM stands for Functional Transformation Method.
"""
import datetime
import h5py
import taslp23
import numpy as np
import os
import pandas as pd
import pnp_synth
from pnp_synth.physical import ftm
import random
import sys
import soundfile as sf
import time
import torch

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/taslp23_fast_test"
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

for module in [h5py, np, pd]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory for audio files.
audio_dir = os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
logscale = True  # the csv files are storing logscaled parameters

synth_type = "ftm"  # Default synthesis type for TASLP23
THETA_COLUMNS = ["omega", "tau", "p", "D", "alpha"]  # FTM parameters

# Batch size for parallel processing
BATCH_SIZE = 100

def synthesize_batch(theta_batch: torch.Tensor, logscale: bool = True) -> torch.Tensor:
    """Synthesize a batch of drum sounds in parallel using MPS."""
    batch_size = theta_batch.shape[0]
    audio_list = []

    # Move batch to device
    theta_batch = theta_batch.to(device)

    # Process each sample in the batch (could potentially be vectorized further)
    for i in range(batch_size):
        x = ftm.rectangular_drum(theta_batch[i], logscale, **ftm.constants)
        audio_list.append(x.cpu())

    return torch.stack(audio_list)

for fold in taslp23.FOLDS:
    # Define path to HDF5 file
    fold_df = taslp23.load_fold(synth_type, fold)
    h5_name = "taslp23_{}_audio.h5".format(fold)
    h5_path = os.path.join(audio_dir, h5_name)

    # Create HDF5 file
    with h5py.File(h5_path, "w") as h5_file:
        audio_group = h5_file.create_group("x")
        shape_group = h5_file.create_group("theta")

    # Process in batches
    total_samples = len(fold_df)
    n_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {fold} fold: {total_samples} samples in {n_batches} batches")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_samples)
        batch_df = fold_df.iloc[start_idx:end_idx]

        # Prepare batch data
        theta_batch = []
        ids_batch = []

        for _, row in batch_df.iterrows():
            theta = np.array([row[column] for column in THETA_COLUMNS])
            theta_batch.append(theta)
            ids_batch.append(str(row["ID"]))

        theta_batch = torch.tensor(np.array(theta_batch), dtype=torch.float32)

        # Synthesize batch
        audio_batch = synthesize_batch(theta_batch, logscale)

        # Write batch to HDF5 (single write operation)
        with h5py.File(h5_path, "a") as h5_file:
            for i, (audio, theta_single, key) in enumerate(zip(audio_batch, theta_batch.cpu().numpy(), ids_batch)):
                h5_file["x"][key] = audio
                h5_file["theta"][key] = theta_single

        # Progress update
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            progress = (batch_idx + 1) / n_batches * 100
            print(f"  Batch {batch_idx + 1}/{n_batches} ({progress:.1f}%)")
            sys.stdout.flush()

    # Print completion
    now = str(datetime.datetime.now())
    print(f"{now} Completed fold: {fold}")
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
