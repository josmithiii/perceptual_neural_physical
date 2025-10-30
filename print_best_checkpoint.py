#!/usr/bin/env .venv/bin/python
"""
Print per-parameter MAE statistics for the best checkpoint.

Usage:
    python print_best_checkpoint.py [model_dir] [epoch]

    If no arguments provided, uses default ICASSP25 P-loss model and finds best checkpoint.
"""
import numpy as np
import sys
import os
import glob

# Add icassp25 to path
sys.path.insert(0, 'icassp25')
import icassp25


def find_best_checkpoint(model_dir: str) -> tuple[str, str]:
    """Find the best checkpoint file with available predictions."""
    # Look for prediction files
    pred_files = glob.glob(os.path.join(model_dir, "test_predictions_epoch*.npy"))

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found in {model_dir}\n"
            f"Run evaluation first: make ei25pl"
        )

    # Extract epoch numbers from prediction files
    available_epochs = []
    for pred_file in pred_files:
        basename = os.path.basename(pred_file)
        # Format: test_predictions_epoch07.npy
        epoch_str = basename.replace("test_predictions_epoch", "").replace(".npy", "")
        available_epochs.append((int(epoch_str), epoch_str, pred_file))

    # Sort by epoch number (descending) to get most recent
    available_epochs.sort(reverse=True)
    epoch_num, epoch_str, pred_file = available_epochs[0]

    print(f"Found {len(available_epochs)} evaluated epoch(s): {[e[1] for e in available_epochs]}")
    print(f"Using most recent: epoch {epoch_str}")

    return pred_file, epoch_str


def compute_statistics(gt_scaled: np.ndarray, pred_scaled: np.ndarray,
                       logscale: int = 1) -> dict:
    """Compute MAE statistics in different spaces."""
    # Inverse transform to physical space
    _, scaler = icassp25.scale_theta(logscale=logscale)
    gt_phys = scaler.inverse_transform(gt_scaled)
    pred_phys = scaler.inverse_transform(pred_scaled)

    # MAE in normalized space
    mae_normalized = np.mean(np.abs(gt_scaled - pred_scaled), axis=0)

    # MAE in physical/log space
    mae_physical = np.mean(np.abs(gt_phys - pred_phys), axis=0)

    # Relative errors in linear space (for frequency parameters)
    relative_errors = {}
    for i, name in enumerate(['omega', 'tau', 'p', 'D']):
        rel_err = np.abs(np.exp(pred_phys[:,i]) - np.exp(gt_phys[:,i])) / np.exp(gt_phys[:,i])
        relative_errors[name] = {
            'median': np.median(rel_err) * 100,
            'mean': np.mean(rel_err) * 100
        }

    return {
        'mae_normalized': mae_normalized,
        'mae_physical': mae_physical,
        'relative_errors': relative_errors,
        'n_samples': gt_scaled.shape[0]
    }


def print_statistics(stats: dict, epoch: str):
    """Print formatted statistics."""
    param_names_norm = ['omega', 'tau', 'p', 'D', 'lm']
    param_names_phys = ['log(omega)', 'log(tau)', 'log(p)', 'log(D)', 'lm']

    print(f"\n{'='*70}")
    print(f"Best Checkpoint Statistics (Epoch {epoch})")
    print(f"{'='*70}")
    print(f"\nTest set size: {stats['n_samples']} samples")

    # Normalized space
    print(f"\n{'-'*70}")
    print("Per-parameter MAE in Normalized Space [-1, 1]:")
    print(f"{'-'*70}")
    for name, mae in zip(param_names_norm, stats['mae_normalized']):
        print(f"  {name:8s}: {mae:.6f}")
    print(f"  {'Overall':8s}: {np.mean(stats['mae_normalized']):.6f}")

    # Physical space
    print(f"\n{'-'*70}")
    print("Per-parameter MAE in Log-Scaled Physical Space:")
    print(f"{'-'*70}")
    for name, mae in zip(param_names_phys, stats['mae_physical']):
        print(f"  {name:12s}: {mae:.6f}")
    print(f"  {'Overall':12s}: {np.mean(stats['mae_physical']):.6f}")

    # Relative errors
    print(f"\n{'-'*70}")
    print("Relative Errors in Linear Space (%):")
    print(f"{'-'*70}")
    print(f"  {'Parameter':10s}  {'Median':>8s}  {'Mean':>8s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for name in ['omega', 'tau', 'p', 'D']:
        median = stats['relative_errors'][name]['median']
        mean = stats['relative_errors'][name]['mean']
        print(f"  {name:10s}  {median:8.2f}  {mean:8.2f}")

    print(f"\n{'='*70}\n")


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        # Default to ICASSP25 P-loss model
        model_dir = './outputs/icassp25/f_W/b0_ploss_finetuneFalse_log-1_minmax-1_opt-adam_batch_size32_lr-0.001_init-test'

    if len(sys.argv) > 2:
        epoch = sys.argv[2]
        pred_file = os.path.join(model_dir, f"test_predictions_epoch{epoch}.npy")
    else:
        # Find best checkpoint automatically
        pred_file, epoch = find_best_checkpoint(model_dir)

    print(f"\nLoading predictions from: {pred_file}")

    # Load predictions
    data = np.load(pred_file, allow_pickle=True)
    gt_scaled = data[0].reshape(-1, 5)
    pred_scaled = data[1].reshape(-1, 5)

    # Compute statistics
    stats = compute_statistics(gt_scaled, pred_scaled, logscale=1)

    # Print results
    print_statistics(stats, epoch)


if __name__ == "__main__":
    main()
