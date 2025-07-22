"""
Audio Comparison Tool: Target vs Reconstruction

This script generates audio comparisons between:
- Target: Original synthesis from ground truth parameters
- Reconstruction: Synthesis from model-predicted parameters

Usage:
    python audio_comparison.py <model_dir> [--num_samples=10] [--output_dir=./audio_comparison]
"""

import argparse
import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path
from tqdm import tqdm

import icassp25
from pnp_synth.neural import forward
from pnp_synth import utils


def load_ground_truth_parameters(data_dir="./icassp25/data"):
    """Load ground truth parameters from CSV."""
    csv_path = os.path.join(data_dir, "full_param_log.csv")
    df = pd.read_csv(csv_path)
    test_df = df[df['fold'] == 'test'].reset_index(drop=True)
    return test_df


def load_model_predictions(model_dir):
    """Load model predictions from .npy files."""
    pred_files = []
    for file in os.listdir(model_dir):
        if file.startswith("test_predictions_epoch") and file.endswith(".npy"):
            epoch = file.split("epoch")[1].split(".npy")[0]
            pred_files.append((int(epoch), file))

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {model_dir}")

    # Use the latest epoch
    pred_files.sort()
    latest_epoch, latest_file = pred_files[-1]

    pred_path = os.path.join(model_dir, latest_file)
    predictions = np.load(pred_path)
    print(f"Loaded predictions from epoch {latest_epoch}: {predictions.shape}")
    return predictions, latest_epoch


def synthesize_audio(parameters, scaler=None, synth_type="ftm", logscale=True):
    """Synthesize audio from parameters."""
    # Convert to tensor
    if isinstance(parameters, np.ndarray):
        parameters = torch.from_numpy(parameters).float()

    # Apply inverse scaling if needed
    if scaler is not None:
        theta = forward.inverse_scale(parameters, scaler)
    else:
        theta = parameters

    # Synthesize audio
    audio = utils.x_from_theta(theta, synth_type=synth_type, logscale=logscale)

    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    return audio


def compute_parameter_errors(targets, predictions):
    """Compute parameter prediction errors."""
    param_names = ["omega", "tau", "p", "D", "alpha"]
    errors = {}

    for i, param in enumerate(param_names):
        target_vals = targets[:, i]
        pred_vals = predictions[:, i]

        mae = np.mean(np.abs(target_vals - pred_vals))
        mse = np.mean((target_vals - pred_vals) ** 2)

        errors[param] = {"mae": mae, "mse": mse, "rmse": np.sqrt(mse)}

    return errors


def create_audio_comparison(model_dir, num_samples=10, output_dir="./audio_comparison"):
    """Create audio comparison files and analysis."""
    print(f"Creating audio comparison in {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading ground truth parameters...")
    test_df = load_ground_truth_parameters()

    print("Loading model predictions...")
    predictions, epoch = load_model_predictions(model_dir)

    # Load scaler
    print("Loading parameter scaler...")
    _, scaler = icassp25.scale_theta(logscale=1)

    # Limit to requested number of samples
    num_samples = min(num_samples, len(test_df), len(predictions))
    test_df = test_df.head(num_samples)
    # Reshape predictions: take the last epoch and flatten batch dimension
    if len(predictions.shape) == 4:  # (epochs, samples, batch_size, params)
        predictions = predictions[-1]  # Take last epoch: (samples, batch_size, params)
        predictions = predictions.reshape(-1, predictions.shape[-1])  # Flatten: (samples*batch_size, params)
    elif len(predictions.shape) == 3:  # (samples, batch_size, params)
        predictions = predictions.reshape(-1, predictions.shape[-1])  # Flatten: (samples*batch_size, params)

    predictions = predictions[:num_samples]

    print(f"Generating audio for {num_samples} samples...")

    # Extract ground truth parameters (exclude ID and fold columns)
    param_cols = ["omega", "tau", "p", "D", "alpha"]
    target_params = test_df[param_cols].values

    # Compute parameter errors
    errors = compute_parameter_errors(target_params, predictions)

    # Generate audio files
    sample_info = []

    for i in tqdm(range(num_samples), desc="Synthesizing audio"):
        sample_id = test_df.iloc[i]["ID"]

        # Synthesize target audio
        target_audio = synthesize_audio(target_params[i], scaler=scaler)
        target_file = f"sample_{sample_id:03d}_target.wav"
        target_path = os.path.join(output_dir, target_file)
        sf.write(target_path, target_audio, 22050)

        # Synthesize reconstruction audio
        recon_audio = synthesize_audio(predictions[i], scaler=scaler)
        recon_file = f"sample_{sample_id:03d}_reconstruction.wav"
        recon_path = os.path.join(output_dir, recon_file)
        sf.write(recon_path, recon_audio, 22050)

        # Store sample info
        sample_info.append({
            "sample_id": sample_id,
            "target_file": target_file,
            "reconstruction_file": recon_file,
            "target_params": dict(zip(param_cols, target_params[i])),
            "predicted_params": dict(zip(param_cols, predictions[i])),
            "param_errors": {p: abs(target_params[i][j] - predictions[i][j])
                           for j, p in enumerate(param_cols)}
        })

    # Save analysis
    analysis = {
        "model_dir": model_dir,
        "epoch": epoch,
        "num_samples": num_samples,
        "parameter_errors": errors,
        "samples": sample_info
    }

    analysis_path = os.path.join(output_dir, "analysis.json")
    import json
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nAudio comparison complete!")
    print(f"Generated {num_samples * 2} audio files in {output_dir}")
    print(f"Analysis saved to {analysis_path}")

    # Print parameter error summary
    print(f"\nParameter Prediction Errors:")
    for param, error in errors.items():
        print(f"  {param}: MAE={error['mae']:.4f}, RMSE={error['rmse']:.4f}")

    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio comparisons")
    parser.add_argument("model_dir", help="Path to model directory with predictions")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to generate (default: 10)")
    parser.add_argument("--output_dir", default="./audio_comparison",
                       help="Output directory (default: ./audio_comparison)")

    args = parser.parse_args()

    create_audio_comparison(args.model_dir, args.num_samples, args.output_dir)
