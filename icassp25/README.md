# ICASSP25 Directory

This directory contains experiments and scripts for the ICASSP 2025 paper on Perceptual-Neural-Physical (PNP) sound matching. The research focuses on neural parameter estimation for physics-based drum synthesis using the Functional Transformation Method (FTM) with enhanced optimization techniques and gradient analysis.

## Configuration

- **`icassp25.py`** - Main configuration module containing shared parameters, data loading functions, forward operators, and neural network components for ICASSP25 experiments. Defines JTFS parameters (J=13, Q=(12,1), shape=2^16), parameter scaling utilities, FTM synthesis operators, and multi-scale spectral loss implementations. Includes factory functions for PNP forward operators with MinMax scaling and log-space parameter handling.

## Data Pipeline

- **`01_generate_audio.py`** - Generates synthetic drum audio using the Functional Transformation Method (FTM) by solving 4th-order partial differential equations. Creates HDF5 files containing audio waveforms and parameter data for train/validation/test splits. ICASSP25 version processes data from `icassp25/data/full_param_log.csv` with CPU-based synthesis for compatibility.

- **`01_generate_audio_test.py`** - Test version of audio generation script that processes only the first 100 samples per fold. Includes error handling and MPS (Apple Silicon) compatibility fixes. Useful for debugging synthesis pipeline and verifying data generation before running full experiments.

- **`01b_compute_pnp_jacobian.py`** - Computes Joint Time-Frequency Scattering (JTFS) coefficients and their Jacobians with respect to normalized synthesis parameters using forward-mode automatic differentiation. Essential for PNP loss computation. ICASSP25 version uses MinMax scaling and creates HDF5 files with Riemannian metric matrices (M = J^T * J) and eigenvalues for analysis.

## Training Scripts

### Framework Training
- **`00_train.py`** - Comprehensive training framework with both `train()` and `eval()` functions. Supports multiple loss types (P-Loss, PNP), EfficientNet variants (B0-B7), and optimizer choices (Adam, Sophia). Includes adaptive Levenberg-Marquardt scheduling for PNP loss, checkpoint management, and TensorBoard logging. Features cross-platform device detection (CUDA/MPS/CPU).

### Specific Training Implementations
- **`03_train_effnet_ploss.py`** - Trains EfficientNet with basic Parametric Loss (P-Loss) using command-line arguments. Short training runs (10 epochs) for rapid prototyping. Includes MPS compatibility and device-specific acceleration settings.

- **`05_train_effnet_pnploss.py`** - Trains EfficientNet with PNP loss incorporating Riemannian metric weights derived from synthesis Jacobians. Uses adaptive Levenberg-Marquardt scheduling with accelerator/brake mechanisms. Full 70-epoch training with comprehensive checkpoint monitoring and gradient folder management.

## Evaluation Scripts

- **`01_eval_ploss.py`** - Evaluation script for P-Loss trained models using the `doce.eval()` framework. Loads model checkpoints and computes test metrics across multiple evaluation criteria including parameter reconstruction accuracy.

- **`02_eval_pnp.py`** - Evaluation script for PNP-Loss trained models with specialized metric computation for perceptual-neural-physical loss variants. Handles weighted parameter evaluation and Riemannian metric analysis.

## Analysis and Optimization Tools

### Gradient Analysis
- **`06_eval_grad.py`** - Comprehensive gradient analysis tool that evaluates gradient norms, smoothness metrics, and optimization dynamics. Supports both P-Loss and PNP loss evaluation with parameter scaling considerations. Includes step-by-step analysis of weight updates, second moment estimation, and convergence behavior for Adam and Sophia optimizers.

- **`08_eval_grad_finetune.py`** - Gradient analysis for fine-tuning scenarios where training transitions from P-Loss to PNP loss mid-training. Evaluates optimization landscape changes and gradient behavior during loss function transitions. Designed for HPC cluster environments with CUDA-specific optimizations.

### Experimental Framework
- **`doce.py`** - Training and evaluation framework functions used by multiple scripts. Contains the main `train()` and `eval()` functions with comprehensive parameter handling, model initialization, and metrics computation. Supports adaptive batch sizing, cross-platform acceleration, and detailed checkpoint analysis.

- **`grad_clip_doce.py`** - DOCE (Design of Computer Experiments) framework configuration for gradient clipping experiments. Defines experimental plans with factors including optimizers (Adam, Sophia), model variants (B0-B7), loss types (P-Loss, PNP), and parameter scaling options. Used for systematic hyperparameter exploration.

## Audio Comparison and Analysis

- **`audio_comparison.py`** - Audio comparison tool that generates target vs reconstruction audio pairs for perceptual evaluation. Loads model predictions, synthesizes audio from both ground truth and predicted parameters, and computes parameter prediction errors (MAE, MSE, RMSE) across all synthesis parameters. Creates organized output with JSON analysis files.

- **`generate_comparison_html.py`** - Generates interactive HTML interface for audio comparisons. Creates web-based player with side-by-side target/reconstruction audio, parameter tables with color-coded error indicators, and navigation controls. Includes overall error summaries and keyboard shortcuts for efficient listening tests.

## Data Structure

The `data/` directory contains:
- **`full_param_log.csv`** - Complete FTM parameter dataset with train/val/test fold assignments
- HDF5 files generated by pipeline scripts for efficient audio and gradient storage

## Key Features

- **Multi-Platform Support**: CUDA, MPS (Apple Silicon), and CPU compatibility
- **Advanced Optimization**: Sophia optimizer implementation with gradient clipping analysis
- **PNP Loss Implementation**: Riemannian metric weighting using synthesis Jacobians
- **Comprehensive Evaluation**: Parameter error analysis, gradient metrics, and audio comparison tools
- **Interactive Analysis**: Web-based interfaces for perceptual evaluation
- **Systematic Experiments**: DOCE framework for hyperparameter exploration

## Synthesis Parameters

- **FTM**: Functional Transformation Method for drum synthesis (5 parameters: ω, τ, p, D, α)
- **Parameter Scaling**: Log-scale transformation for frequency parameters, MinMax normalization to [-1, 1]
- **Riemannian Metrics**: Physics-informed loss weighting using gradient-derived metric tensors

## Usage Notes

All training scripts accept command-line arguments for flexible configuration including:
- EfficientNet architecture variants (B0-B7)
- Optimizer choice (Adam, Sophia)
- Loss types (P-Loss, PNP)
- Parameter scaling options
- Batch sizes and training epochs
- Trial/initialization IDs for reproducible experiments

The pipeline follows a numbered sequence: data generation (01*) → Jacobian computation (01b*) → training (03*, 05*) → evaluation and analysis (various evaluation scripts).
