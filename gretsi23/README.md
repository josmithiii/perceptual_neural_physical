# GRETSI23 Directory

This directory contains experiments and scripts for the GRETSI 2023 paper on Perceptual-Neural-Physical (PNP) sound matching. The research focuses on neural parameter estimation for physics-based synthesis using amplitude-modulated (AM) chirp signals, providing a complementary study to drum synthesis with different parameter scaling and synthesis approaches.

## Configuration

- **`setups.py`** - Main configuration module containing shared parameters, data loading functions, forward operators, and neural network components for GRETSI23 experiments. Defines JTFS parameters (J=13, Q=(12,1), shape=2^15), AMChirp synthesis parameters (3 parameters: f0, fm, γ), parameter scaling utilities with log-space handling, and comprehensive data generation with geometric parameter sampling. Includes factory functions for PNP forward operators with MinMax scaling to [-1,1].

## Data Pipeline

### Audio Generation
- **`01_generate_audio.py`** - Generates synthetic amplitude-modulated chirp signals using the AMChirp synthesis model. Creates ~27k samples through systematic parameter space exploration using geometric sampling (30³ grid with randomization). Parameters include carrier frequency (f0: 512-1024 Hz), modulation frequency (fm: 4-16 Hz), and chirp rate (γ: 0.5-4). Creates HDF5 files containing audio waveforms and parameter data for train/validation/test splits with intelligent fold assignment (validation set centered in parameter space).

### Gradient and Metric Computation
- **`02_compute_pnp_jacobian.py`** - Computes Joint Time-Frequency Scattering (JTFS) coefficients and their Jacobians with respect to AMChirp synthesis parameters using forward-mode automatic differentiation. Essential for PNP loss computation. Supports both MinMax scaling and raw parameter modes (controlled by `dir_name` configuration). Creates HDF5 files containing metric matrices (M = J^T * J) and eigenvalues for Riemannian weighting in PNP training.

## Training Scripts

The training scripts are organized by loss function and parameter scaling variants, with systematic exploration of log-scale vs raw parameters and different PNP weighting strategies.

### Basic Loss Functions
- **`03_train_effnet_ploss.py`** - Trains EfficientNet with basic Parametric Loss (P-Loss) using MSE between predicted and ground truth AMChirp parameters. Serves as baseline approach with 70 epochs, 3 output dimensions (f0, fm, γ), GPU acceleration, and comprehensive checkpoint management with TensorBoard logging.

- **`03b_train_effnet_ploss_nolog.py`** - P-Loss training variant without log-scale parameter transformation, exploring raw parameter space optimization.

- **`03c_train_effnet_ploss_raw.py`** - P-Loss training using raw (non-MinMax scaled) parameters for direct parameter space optimization without normalization.

- **`03d_train_effnet_ploss_raw_log.py`** - P-Loss training combining raw parameter handling with log-scale transformation for frequency parameters.

### Spectral Loss Training
- **`06_train_effnet_specloss.py`** - Trains EfficientNet with Multi-Scale Spectral Loss (MSS) using STFT representations across multiple time-frequency resolutions. Focuses on perceptual reconstruction fidelity of AMChirp signals with 70 epochs and optimized hyperparameters for tonal synthesis.

- **`06c_train_effnet_specloss_raw.py`** - MSS training variant using raw (non-scaled) parameters for direct optimization in the original parameter space.

- **`06d_train_effnet_specloss_log.py`** - MSS training with log-scale parameter transformation for improved optimization of frequency-based parameters.

### PNP Loss Training - Basic Implementations
- **`05_train_effnet_pnploss_novol.py`** - PNP training without Riemannian volume weights (`weight_type="novol"`) using constant Levenberg-Marquardt scheduling. Uses `LMA={'mode': 'constant', 'lambda': 1e-8}` with identity damping for stable AMChirp parameter estimation with 70 epochs.

- **`05b_train_effnet_pnploss_novol_nolog.py`** - PNP "novol" variant without log-scale transformation, exploring PNP loss effectiveness in raw parameter space.

- **`05c_train_effnet_pnploss_novol_raw.py`** - PNP training using raw parameters without MinMax normalization, maintaining physics-based parameter ranges.

- **`05d_train_effnet_pnploss_novol_log.py`** - PNP training combining "novol" weighting with log-scale parameter transformation for frequency parameters.

### Advanced PNP Loss Variants

#### Adaptive PNP Training
- **`04_train_effnet_pnploss_novol_adaptive_brake1.py`** - PNP training with adaptive Levenberg-Marquardt scheduling using dynamic lambda adjustment based on training progress. Implements accelerator/brake mechanisms for robust convergence in AMChirp parameter space.

#### Mean Metric PNP Training
- **`07_train_effnet_pnploss_novol_meanM.py`** - Advanced PNP training using mean Riemannian metrics (`damping="mean"`) computed across the dataset. Uses constant LMA mode with `lambda=1` for enhanced physics-informed weighting based on average gradient behavior.

- **`07b_train_effnet_pnploss_novol_meanM_nolog.py`** - Mean metric PNP variant without log-scale transformation, testing mean weighting effectiveness in raw parameter space.

- **`07c_train_effnet_pnploss_novol_meanM_raw.py`** - Mean metric PNP training using raw parameters for direct optimization with averaged Riemannian weighting.

- **`07d_train_effnet_pnploss_novol_meanM_log.py`** - Mean metric PNP training with log-scale transformation, combining averaged physics-informed weighting with optimized parameter scaling.

## Key Features

### AMChirp Synthesis Model
- **3-Parameter System**: Carrier frequency (f0), modulation frequency (fm), chirp rate (γ)
- **Geometric Parameter Sampling**: Logarithmic spacing across parameter ranges for comprehensive coverage
- **Physics-Based Synthesis**: Amplitude-modulated chirp generation with differentiable synthesis pipeline

### Parameter Scaling Exploration
- **Log-Scale vs Raw**: Systematic comparison of log-transformed vs direct parameter optimization
- **MinMax vs Raw**: Exploration of normalized [-1,1] vs physics-based parameter ranges
- **Naming Convention**: Scripts use suffixes (_nolog, _raw, _log) to indicate parameter handling approach

### Loss Function Hierarchy
- **P-Loss**: Basic parameter MSE for AMChirp parameters (baseline)
- **Spectral Loss**: Multi-scale STFT reconstruction optimized for tonal signals
- **PNP Loss**: Perceptual-Neural-Physical with Riemannian metric weighting from AMChirp synthesis Jacobians

### Advanced Optimization Techniques
- **Levenberg-Marquardt Variants**: Constant vs adaptive scheduling for AMChirp parameter estimation
- **Mean Metric Damping**: Dataset-averaged Riemannian weighting for improved stability
- **Physics-Informed Weighting**: Gradient-derived metric tensors specific to AMChirp synthesis

### Cross-Platform Support
- **GPU Acceleration**: CUDA-optimized training with automatic device detection
- **Efficient Audio Processing**: HDF5-based storage and batch processing for 27k samples
- **TensorBoard Integration**: Comprehensive logging and visualization for experiment tracking

## Data Structure

The `data/` directory contains:
- **`full_param_log.csv`** - Complete AMChirp parameter dataset (27k samples) with train/val/test fold assignments based on geometric sampling and intelligent validation set placement

## Synthesis Parameters

- **AMChirp**: Amplitude-modulated chirp synthesis (3 parameters: f0, fm, γ)
- **Parameter Ranges**:
  - f0 (carrier): 512-1024 Hz (log-scale)
  - fm (modulation): 4-16 Hz (log-scale)
  - γ (chirp rate): 0.5-4 (log-scale)
- **Riemannian Metrics**: Physics-informed loss weighting using M = J^T * J from AMChirp synthesis Jacobians

## Usage Notes

All training scripts accept command-line arguments for flexible configuration:
- `save_dir`: Data directory path
- `init_id`: Experiment initialization/trial ID
- `batch_size`: Training batch size
- `ckpt_path`: Optional checkpoint path for resuming training

The pipeline follows the same numbered sequence as other experiment directories:
1. **Data Generation** (`01_generate_audio.py`) → **Gradient Computation** (`02_compute_pnp_jacobian.py`)
2. **Training** (`03*-07*_train_effnet_*.py`) with systematic exploration of parameter scaling approaches

## Research Focus

This directory explores PNP methodology applied to **tonal synthesis** (AMChirp) rather than percussive synthesis (FTM drums), providing insights into:
- **Parameter scaling effects** on neural parameter estimation
- **PNP loss effectiveness** across different synthesis models
- **Riemannian metric behavior** for tonal vs percussive signals
- **Optimization landscape differences** between 3-parameter (AMChirp) and 5-parameter (FTM) systems

The systematic exploration of log-scale vs raw parameter handling provides crucial insights for physics-informed neural audio synthesis across different signal types and parameter dimensionalities.
