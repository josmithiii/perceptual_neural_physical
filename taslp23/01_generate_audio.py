"""
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
from pnp_synth.synth_registry import get_synth_config, get_synthesis_function, validate_synth_type
import random
import sys
import soundfile as sf
import time
import torch

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
save_dir = sys.argv[1]
synth_type = sys.argv[2] if len(sys.argv) > 2 else "ftm"  # Default to FTM for backward compatibility
validate_synth_type(synth_type)
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")
print(f"Using synthesizer type: {synth_type}\n")

for module in [h5py, np, pd]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Create directory for audio files.
audio_dir = os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
logscale = True #the csv files are storing logscaled parameters

# Get synthesizer configuration
config = get_synth_config(synth_type)
THETA_COLUMNS = config["theta_columns"]
synth_fn, constants = get_synthesis_function(synth_type)
print(f"Parameter columns: {THETA_COLUMNS}")
print(f"Synthesis function: {config['synthesis_fn']}\n")

for fold in taslp23.FOLDS:
    # Define path to HDF5 file
    fold_df = taslp23.load_fold(synth_type, fold)
    h5_name = "{}_{}_audio.h5".format(synth_type, fold)
    h5_path = os.path.join(audio_dir, h5_name)

    # Create HDF5 file
    with h5py.File(h5_path, "w") as h5_file:
        audio_group = h5_file.create_group("x")
        shape_group = h5_file.create_group("theta")

    # Define row iterator
    row_iter = fold_df.iterrows()

    # Loop over batches.
    batch_size = len(fold_df)
    n_batches = 1 + len(fold_df) // batch_size

    for i, row in row_iter:
        #i, row = irow

        # Physical audio synthesis (g). theta -> x
        theta = np.array([row[column] for column in THETA_COLUMNS])
        theta_tensor = torch.tensor(theta, dtype=torch.float32)

        # Dynamic synthesis call based on synth_type
        if synth_type == "string":
            x = synth_fn(theta_tensor, 0.1, **constants)  # pos_ratio=0.1 for string
        else:
            x = synth_fn(theta_tensor, logscale, **constants)
        key = str(row["ID"])

        # Append to HDF5 file
        with h5py.File(h5_path, "a") as h5_file:
            # Store shape annd waveform into HDF5 container.
            h5_file["x"][key] = x.cpu()
            h5_file["theta"][key] = theta

    # Print
    now = str(datetime.datetime.now())
    #batch_str = str(batch_id).zfill(len(str(n_batches)))
    #print(now + " Exported: {}, batch {}".format(fold, batch_str))
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
