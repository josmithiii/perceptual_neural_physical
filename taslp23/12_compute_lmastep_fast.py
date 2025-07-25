"""
FAST VERSION using MPS acceleration, batch processing, and parallel I/O.
This script calculates the M matrices and sigmas corresponding to each parameter
using optimized computation for MacBook Pro (16 CPU cores, 40 GPU cores, 128GB RAM).
"""
import datetime
import functools
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import h5py
import numpy as np
import pandas as pd
import sklearn
import torch

try:
    from torch.func import jacfwd
except ImportError:
    from functorch import jacfwd

import kymatio
import taslp23
import pnp_synth


def setup_device_and_precision(force_cpu=False):
    """Setup optimal device and precision for computation."""
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

    return device, precision


def create_h5_files(save_dir, dir_name, synth_type, id_start):
    """Create H5 files for all folds with proper structure."""
    os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)

    for fold in taslp23.FOLDS:
        h5_name = f"{synth_type}_{fold}_M_{id_start}.h5"
        h5_path = os.path.join(save_dir, dir_name, h5_name)
        if not os.path.exists(h5_path):
            with h5py.File(h5_path, "w") as h5_file:
                h5_file.create_group("M")
                h5_file.create_group("sigma")
                h5_file.create_group("JdagJ")  # Add this for compatibility


def check_existing_computation(save_dir, dir_name, synth_type, sample_id, fold, id_start):
    """Check if computation already exists for this sample."""
    h5_name = f"{synth_type}_{fold}_M_{id_start}.h5"
    h5_path = os.path.join(save_dir, dir_name, h5_name)
    try:
        with h5py.File(h5_path, "r") as h5_file:
            return str(sample_id) in h5_file['sigma'].keys()
    except (KeyError, OSError):
        return False


def save_results_batch(results_batch, save_dir, dir_name, synth_type, id_start):
    """Save a batch of results to H5 files with thread safety."""
    if not results_batch:
        return

    # Group results by fold for efficient file access
    fold_groups = {}
    for sample_id, fold, M_cpu, sigma_cpu, J_cpu in results_batch:
        if fold not in fold_groups:
            fold_groups[fold] = []
        fold_groups[fold].append((sample_id, M_cpu, sigma_cpu, J_cpu))

    # Save each fold's results
    for fold, fold_results in fold_groups.items():
        h5_name = f"{synth_type}_{fold}_M_{id_start}.h5"
        h5_path = os.path.join(save_dir, dir_name, h5_name)

        with h5py.File(h5_path, "a") as h5_file:
            for sample_id, M_cpu, sigma_cpu, J_cpu in fold_results:
                # Check if already exists before writing (eliminates race condition)
                if str(sample_id) in h5_file['sigma']:
                    print(f"Sample {sample_id} already computed, skipping...")
                    continue
                h5_file['M'][str(sample_id)] = M_cpu.numpy()
                h5_file['sigma'][str(sample_id)] = sigma_cpu.numpy()
                if J_cpu is not None:
                    # Compute JdagJ = J.T @ J on CPU to save memory
                    JdagJ = torch.matmul(J_cpu.T, J_cpu)
                    h5_file['JdagJ'][str(sample_id)] = JdagJ.numpy()


def process_sample_batch(batch_data, dS_over_dnu, device, precision):
    """Process a batch of samples on GPU/MPS device."""
    batch_indices, nus_batch, fold_batch = batch_data
    results = []

    try:
        # Move batch to device
        device_nus = []
        for nu_sample in nus_batch:
            nu_tensor = torch.tensor(nu_sample, requires_grad=True, device=device, dtype=precision)
            device_nus.append(nu_tensor)

        # Process each sample in the batch
        for idx, nu_tensor in enumerate(device_nus):
            sample_id = batch_indices[idx]
            fold = fold_batch[idx]

            try:
                # Compute Jacobian: d(S) / d(nu)
                J = dS_over_dnu(nu_tensor).detach()

                # Compute M = J.T @ J (Riemannian metric)
                M = torch.matmul(J.T, J)
                assert M.shape[0] == 5 and M.shape[1] == 5, f"Expected M shape (5,5), got {M.shape}"

                # Compute eigenvalues
                sigma = torch.linalg.eigvals(M)

                # Move results to CPU for storage
                M_cpu = M.detach().cpu()
                sigma_cpu = sigma.detach().cpu()
                J_cpu = J.detach().cpu()  # Keep full Jacobian for JdagJ computation

                results.append((sample_id, fold, M_cpu, sigma_cpu, J_cpu))

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                    print(f"GPU memory error for sample {sample_id}: {e}")
                    # Try to continue with remaining samples
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                    continue
                else:
                    print(f"Runtime error for sample {sample_id}: {e}")
                    continue
            except Exception as e:
                print(f"Unexpected error for sample {sample_id}: {e}")
                continue

    except Exception as e:
        print(f"Batch processing error: {e}")
        return []

    return results


def main():
    # Print header
    start_time = int(time.time())
    print(str(datetime.datetime.now()) + " Start.")
    print(__doc__ + "\n")

    # Parse command line arguments
    save_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs/taslp23"
    id_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    id_end = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    force_cpu = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # Optional: force CPU for float64

    print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

    # Print version information
    for module in [kymatio, np, pd, sklearn, torch]:
        print(f"{module.__name__} version: {module.__version__}")
    print("")
    sys.stdout.flush()

    # Configuration
    dir_name = "M_log"
    synth_type = "ftm"
    THETA_COLUMNS = ["omega", "tau", "p", "D", "alpha"]

    # Setup device and precision
    device, precision = setup_device_and_precision(force_cpu)

    # Determine optimal batch size based on device and available memory
    if device.type == "mps":
        BATCH_SIZE = 200  # was 50
    elif device.type == "cuda":
        BATCH_SIZE = 100
    else:
        BATCH_SIZE = 25  # CPU processing

    print(f"Using batch size: {BATCH_SIZE}")

    # Load and prepare data
    if "log" in dir_name:
        logscale = True
    else:
        logscale = False

    if "nominmax" in dir_name:
        scaler = None
        full_df = taslp23.load_fold(synth_type)
        nus = []
        for column in THETA_COLUMNS:
            if not logscale and column in ["omega", "p", "D"]:
                nus.append(10 ** full_df[column].values)
            else:
                nus.append(full_df[column].values)
        nus = np.stack(nus, axis=1)
    else:
        _, scaler = taslp23.scale_theta(logscale, synth_type)
        full_df = taslp23.load_fold(synth_type)
        nus = []
        for column in THETA_COLUMNS:
            if not logscale and column in ["omega", "p", "D"]:
                nus.append(10 ** full_df[column].values)
            else:
                nus.append(full_df[column].values)
        nus = np.stack(nus, axis=1)

    # Create H5 files
    create_h5_files(save_dir, dir_name, synth_type, id_start)

    # Define the forward PNP operator
    S_from_nu = taslp23.pnp_forward_factory(scaler, logscale, synth_type)

    # Define the associated Jacobian operator
    dS_over_dnu = jacfwd(S_from_nu)

    # Prepare batch data
    total_samples = id_end - id_start
    n_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {total_samples} samples in {n_batches} batches of size {BATCH_SIZE}")
    sys.stdout.flush()

    # Filter samples that need computation
    samples_to_process = []
    for i in range(id_start, id_end):
        row = full_df.iloc[i]
        key = int(row["ID"])
        fold = row['fold']

        if not check_existing_computation(save_dir, dir_name, synth_type, i, fold, id_start):
            samples_to_process.append((i, key, fold))

    print(f"Found {len(samples_to_process)} samples that need computation (skipping {total_samples - len(samples_to_process)} existing)")

    if not samples_to_process:
        print("All computations already exist. Exiting.")
        return

    # Process in batches with parallel I/O
    torch.autograd.set_detect_anomaly(True)

    # Use ThreadPoolExecutor for I/O operations
    with ThreadPoolExecutor(max_workers=8) as io_executor:
        processed_count = 0

        for batch_start in range(0, len(samples_to_process), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(samples_to_process))
            batch_samples = samples_to_process[batch_start:batch_end]

            # Prepare batch data
            batch_indices = [sample[0] for sample in batch_samples]
            batch_keys = [sample[1] for sample in batch_samples]
            batch_folds = [sample[2] for sample in batch_samples]
            nus_batch = [nus[key] for key in batch_keys]

            batch_data = (batch_indices, nus_batch, batch_folds)

            # Process batch on GPU/MPS
            results_batch = process_sample_batch(batch_data, dS_over_dnu, device, precision)

            if results_batch:
                # Submit I/O operation to thread pool
                io_future = io_executor.submit(
                    save_results_batch,
                    results_batch,
                    save_dir,
                    dir_name,
                    synth_type,
                    id_start
                )

                processed_count += len(results_batch)

                # Progress update
                progress = processed_count / len(samples_to_process) * 100
                print(f"Progress: {processed_count}/{len(samples_to_process)} samples ({progress:.1f}%)")

                # Report successful samples
                for sample_id, fold, _, _, _ in results_batch:
                    now = str(datetime.datetime.now())
                    print(f"{now} Computed: {fold}/{synth_type}_{sample_id:06d}")

                sys.stdout.flush()

    print("")

    # Print elapsed time
    print(str(datetime.datetime.now()) + " Success.")
    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.0
    elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:05.2f}"
    print(f"Total elapsed time: {elapsed_str}.")


if __name__ == "__main__":
    main()
