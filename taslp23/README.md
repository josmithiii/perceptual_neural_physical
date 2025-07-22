# TASLP23 Directory

This directory contains experiments and scripts for the TASLP 2023 paper on Perceptual-Neural-Physical (PNP) sound matching. The research focuses on neural parameter estimation for physics-based drum synthesis using the Functional Transformation Method (FTM).

## Configuration

- **`taslp23.py`** - Main configuration module containing shared parameters, data loading functions, forward operators, and neural network components for the TASLP23 experiments. Defines JTFS parameters, parameter scaling, synthesis operators, and multi-scale spectral loss implementations.

## Data Pipeline

- **`01_generate_audio.py`** - Generates synthetic drum audio using the Functional Transformation Method (FTM) by solving 4th-order partial differential equations. Creates HDF5 files containing audio waveforms and parameter data for train/validation/test splits.

- **`02_compute_pnp_jacobian.py`** - Computes Joint Time-Frequency Scattering (JTFS) coefficients and their Jacobians with respect to normalized synthesis parameters using automatic differentiation. Essential for PNP loss computation.

## Training Scripts

### Basic Training
- **`03_train_effnet_ploss.py`** - Trains EfficientNet with basic Parametric Loss (P-Loss) for drum sound parameter estimation. Serves as baseline approach.

- **`04_train_effnet_specloss.py`** - Trains EfficientNet with Multi-Scale Spectral Loss (MSS) using STFT representations across multiple time-frequency resolutions.

### PNP Training
- **`05_train_effnet_pnploss.py`** - Trains EfficientNet with PNP loss incorporating Riemannian metric weights derived from synthesis Jacobians. Uses adaptive Levenberg-Marquardt scheduling.

- **`08_train_effnet_pnploss_raw.py`** - PNP training variant using raw (non-MinMax scaled) parameters with different Jacobian folder structure.

### Fine-tuning Approaches
- **`06_train_effnet_finetune_pnploss.py`** - Two-stage training: first with P-Loss, then fine-tuning with PNP loss. Loads pretrained models and continues optimization.

- **`07_train_effnet_finetune_mss.py`** - Two-stage training: first with P-Loss, then fine-tuning with Multi-Scale Spectral loss for improved perceptual matching.

- **`09_train_effnet_finetune_pnploss.py`** - Alternative PNP fine-tuning implementation with different hyperparameters and checkpoint handling.

- **`10_train_effnet_finetune_mss.py`** - Alternative MSS fine-tuning implementation with modified batch sizes and training configurations.

## Utilities

- **`12_compute_lmastep.py`** - Computes Riemannian metric matrices (M) and eigenvalues (sigma) from synthesis Jacobians for use in PNP loss weighting. Creates HDF5 files with precomputed metrics.

- **`13_merge_h5file.py`** - Merges multiple HDF5 files containing metric matrices and eigenvalues into consolidated files per data fold (train/val/test).

## Data Structure

The `data/` directory contains:
- **`ftm/`** - FTM synthesis parameter logs
  - `full_param_log.csv` - Complete parameter dataset
  - `train_param_log.csv`, `val_param_log.csv`, `test_param_log.csv` - Data splits
- **`amchirp/`** - Alternative synthesis model data (if used)

## SLURM Batch Scripts

The `sbatch/` directory contains cluster job submission scripts for parallel training on SLURM-based HPC systems.

## Key Features

- **Multi-loss Training**: Supports P-Loss, Spectral Loss, and PNP Loss
- **Parameter Scaling**: Log-scale and MinMax normalization options
- **Riemannian Metrics**: Physics-informed weighting using synthesis Jacobians
- **Fine-tuning**: Two-stage training approaches for improved convergence
- **Cluster Support**: SLURM batch scripts for HPC deployment

## Synthesis Types

- **FTM**: Functional Transformation Method for drum synthesis (5 parameters: ω, τ, p, D, α)
- **AMChirp**: Amplitude-modulated chirp synthesis (3 parameters: f0, fm, γ)

All scripts accept command-line arguments for flexible configuration including synthesis type, parameter scaling, optimization settings, and batch sizes.
