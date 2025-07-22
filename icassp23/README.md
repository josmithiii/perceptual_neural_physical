# ICASSP23 Directory

This directory contains experiments and scripts for the ICASSP 2023 paper on Perceptual-Neural-Physical (PNP) sound matching. The research focuses on neural parameter estimation for physics-based drum synthesis using the Functional Transformation Method (FTM) with comprehensive exploration of loss functions and optimization strategies.

## Configuration

- **`icassp23.py`** - Main configuration module containing shared parameters, data loading functions, forward operators, and neural network components for ICASSP23 experiments. Defines JTFS parameters (J=13, Q=(12,1), shape=2^16), FTM synthesis parameters, parameter scaling utilities, and Multi-Scale Spectral Loss implementations. Includes factory functions for PNP forward operators with MinMax scaling to [-1,1] and log-scale parameter handling.

## Data Pipeline

### Audio Generation
- **`01_generate_audio.py`** - Generates synthetic drum audio using the Functional Transformation Method (FTM) by solving 4th-order partial differential equations. Creates HDF5 files containing audio waveforms and parameter data for train/validation/test splits (5 parameters: ω, τ, p, D, α). Uses log-scale parameters and CPU-based synthesis with comprehensive error handling.

### Gradient and Metric Computation
- **`02_compute_pnp_jacobian.py`** - Computes Joint Time-Frequency Scattering (JTFS) coefficients and their Jacobians with respect to normalized synthesis parameters using forward-mode automatic differentiation (`functorch.jacfwd`). Essential for PNP loss computation. Processes data in batches specified by command-line arguments (id_start, id_end) and saves results as NumPy files.

- **`05_compute_M.py`** - Computes Riemannian metric matrices M(θ) = J(θ)^T * J(θ) from synthesis Jacobians. Creates HDF5 files containing metric matrices and eigenvalues organized by data folds. Essential preprocessing for physics-informed loss weighting in PNP training.

- **`13_compute_lmastep.py`** - Advanced gradient computation script that calculates JTFS coefficients and Levenberg-Marquardt Algorithm (LMA) steps using Multi-Scale Spectral (MSS) features instead of JTFS. Computes Jacobians (J), pseudo-inverses (J†), metric matrices (M), and eigenvalues (σ) for modal analysis. Supports both MinMax scaling and log-scale parameter modes.

### Data Management
- **`02b_write_h5files.py`** - Utility script for writing and organizing HDF5 files in the data pipeline.

- **`14_merge_h5file.py`** - Merges multiple HDF5 files containing metric matrices and eigenvalues into consolidated files per data fold. Combines distributed computation results from cluster jobs into unified datasets for training.

## Training Scripts

### Basic Loss Functions
- **`03_train_effnet_ploss.py`** - Trains EfficientNet with basic Parametric Loss (P-Loss) using MSE between predicted and ground truth parameters. Serves as baseline approach with 70 epochs, supports MPS (Apple Silicon) and CUDA acceleration, includes comprehensive checkpoint management and TensorBoard logging.

- **`03b_train_effnet_ploss.py`** - Alternative P-Loss training implementation with different hyperparameters and configuration options.

- **`03c_train_effnet_ploss_modal_nominmax.py`** - P-Loss training variant using modal analysis without MinMax scaling for parameter normalization.

### Spectral Loss Training
- **`04_train_effnet_specloss.py`** - Trains EfficientNet with Multi-Scale Spectral Loss (MSS) using STFT representations across multiple time-frequency resolutions. Focuses on perceptual reconstruction fidelity with 70 epochs and GPU-accelerated training.

- **`04b_train_effnet_specloss.py`** - Alternative MSS training implementation with modified hyperparameters and batch configurations.

### PNP Loss Training - Basic Implementations
- **`06_train_effnet_pnploss.py`** - Trains EfficientNet with full PNP loss incorporating Riemannian metric weights derived from synthesis Jacobians. Uses `weight_type="pnp"` with complete volume weighting, 30 epochs, and adaptive learning rate scheduling. Includes cross-platform device detection (CUDA/MPS/CPU).

- **`07_train_effnet_pnploss_novol.py`** - PNP training without Riemannian volume weights (`weight_type="novol"`). Uses adaptive Levenberg-Marquardt scheduling with configurable accelerator/brake mechanisms. Extended training (70 epochs) with comprehensive checkpoint monitoring.

- **`07b_train_effnet_pnploss_novol.py`** - Alternative "novol" PNP implementation with different LMA parameters and training configurations.

### Adaptive PNP Loss Variants
- **`08_train_effnet_pnploss_novol_adaptive_brake1.py`** - PNP training with adaptive LMA scheduling using `mode="adaptive"` with fine-tuned brake parameters (brake=10, accelerator=0.1). Uses identity damping and high initial lambda (1e+20) for robust convergence.

- **`08b_train_effnet_pnploss_novol_adaptive_brake1.py`** - Alternative adaptive implementation with modified brake settings (brake=1, accelerator=0.2) for different convergence characteristics.

- **`11_train_effnet_pnploss_novol_adaptive_brake1_diag.py`** - Diagnostic version of adaptive PNP training with enhanced logging and analysis capabilities for optimization behavior study.

### Scheduled PNP Loss Variants
- **`09_train_effnet_pnploss_novol_scheduled.py`** - PNP training with scheduled LMA parameter updates following predetermined timeline rather than adaptive adjustment. Provides deterministic training progression for reproducible experiments.

- **`09b_train_effnet_pnploss_novol_scheduled.py`** - Alternative scheduled implementation with different scheduling parameters and validation strategies.

- **`10_train_effnet_pnploss_novol_const.py`** - PNP training with constant LMA parameters throughout training (`mode="constant"`). Eliminates adaptive behavior for controlled experimental conditions.

### Modal Analysis Training
- **`13b_train_effnet_pnploss_modal_meanM_log.py`** - Advanced PNP training using modal analysis with mean metric matrices and log-scale parameters. Incorporates eigenvalue analysis for enhanced physics-informed weighting.

- **`13e_train_effnet_pnploss_modal_meanM_log_interpol.py`** - Modal PNP training with interpolation techniques for smooth parameter space exploration and improved convergence stability.

## Key Features

### Loss Function Hierarchy
- **P-Loss**: Basic parameter MSE (baseline approach)
- **Spectral Loss**: Multi-scale STFT reconstruction with perceptual weighting
- **PNP Loss**: Perceptual-Neural-Physical with Riemannian metric weighting from synthesis Jacobians

### Advanced Optimization Techniques
- **Levenberg-Marquardt Adaptation**: Multiple scheduling strategies (adaptive, scheduled, constant)
- **Riemannian Metrics**: Physics-informed loss weighting using gradient-derived metric tensors
- **Modal Analysis**: Eigenvalue-based parameter space analysis for enhanced training stability

### Parameter Scaling Strategy
- **Log-scale transformation** for frequency parameters (ω, p, D)
- **MinMax normalization** to [-1, 1] for neural network compatibility
- **Differentiable inverse scaling** in synthesis forward pass

### Cross-Platform Support
- **CUDA**: Full GPU acceleration for NVIDIA hardware
- **MPS**: Apple Silicon GPU compatibility with float32 precision
- **CPU**: Fallback support with optimized threading

## Data Structure

The `data/` directory contains:
- **`full_param_log.csv`** - Complete FTM parameter dataset with train/val/test fold assignments
- **`train_param_log.csv`, `val_param_log.csv`, `test_param_log.csv`** - Pre-split data folds
- HDF5 files generated by pipeline scripts for efficient audio and gradient storage

## SLURM Batch Scripts

The `sbatch/` directory contains comprehensive cluster job submission scripts for parallel training and computation on SLURM-based HPC systems with hundreds of individual job files for distributed processing.

## Jupyter Notebooks

- **`Compare trained models.ipynb`** - Comparative analysis of different training approaches and loss functions
- **`Joint time-frequency scattering.ipynb`** - JTFS feature analysis and visualization
- **`Visualize generated samples.ipynb`** - Audio synthesis visualization and parameter space exploration

## Synthesis Parameters

- **FTM**: Functional Transformation Method for drum synthesis (5 parameters: ω, τ, p, D, α)
- **Parameter Ranges**: Log-scale for frequency parameters, normalized ranges for physical parameters
- **Riemannian Metrics**: Physics-informed loss weighting using M = J^T * J from synthesis Jacobians

## Usage Notes

All training scripts accept command-line arguments for flexible configuration:
- `save_dir`: Data directory path
- `init_id`: Experiment initialization/trial ID
- `batch_size`: Training batch size (varies by loss type)
- `ckpt_path`: Optional checkpoint path for resuming training

The pipeline follows a numbered sequence:
1. **Data Generation** (`01_generate_audio.py`) → **Gradient Computation** (`02_compute_pnp_jacobian.py`)
2. **Metric Computation** (`05_compute_M.py`) → **Training** (`03*-13*_train_effnet_*.py`)
3. **Data Consolidation** (`14_merge_h5file.py`) for distributed results

This directory represents the foundational ICASSP23 research establishing core PNP methodology with comprehensive exploration of loss functions, optimization strategies, and parameter scaling approaches for physics-informed neural audio synthesis.
