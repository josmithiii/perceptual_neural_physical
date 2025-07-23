# MERSENNE24 Directory

This directory contains experiments for Mersenne 2024 research on noise-robust neural parameter estimation for physics-based drum synthesis. The research focuses on training EfficientNet models with various noise augmentation strategies to improve robustness of parameter estimation from real-world drum recordings mixed with synthetic physics-based synthesis.

## Configuration

- **`mersenne24.py`** - Main configuration module containing shared parameters, data loading functions, and synthesis operators for Mersenne24 experiments. Defines both JTFS parameters (J=13, Q=(12,1), shape=2^16) and Multi-Scale Spectral Loss parameters for alternative feature extraction. Implements parameter scaling utilities, physics-based synthesis operators for string/drum models, and factory functions for PNP forward operators with MinMax scaling. Supports both "phys" and "perc" parameter modes with 6-parameter physics-based synthesis (EI, Ts0, d1, d3, lm, ell).

## Data Pipeline

- **`01_generate_audio.py`** - Generates synthetic drum audio using the Functional Transformation Method (FTM) by solving 4th-order partial differential equations for string/drum synthesis. Creates HDF5 files containing audio waveforms and parameter data for train/validation/test splits. Uses perceptual parameter mapping (`ftm.linearstring_percep`) with log-scale parameter handling for compatibility with existing datasets.

- **`01_generate_audio_phys.py`** - Physics-based audio generation variant that uses physical parameter mapping (`ftm.linearstring_physics`) with randomized excitation positions. Adds position ratio randomization for more diverse synthesis and includes error handling for failed synthesis attempts. Creates additional HDF5 group for storing excitation position data alongside audio and parameters.

## Training Scripts - Noise Robustness Experiments

### Baseline Training
- **`03_train_effnet_ploss.py`** - Baseline EfficientNet training with Parametric Loss (P-Loss) using clean synthetic data only. Trains for 70 epochs with batch size 256, supports checkpoint resumption, and evaluates on both synthetic and noise-augmented test sets. Includes comprehensive argument parsing for experimental configuration (optimizer, scaling, parameter modes).

### Noise Augmentation Strategies
- **`04_train_effnet_ploss_matchednoise.py`** - Trains EfficientNet with "matched" noise augmentation where real audio segments are paired with synthetic drum sounds based on similar acoustic characteristics. Uses smaller batch size (64) and 50 epochs with dual checkpoint saving (best and last). Implements `noise_mode="matched"` for acoustically-aligned noise mixing.

- **`04_train_effnet_ploss_noise.py`** - Trains EfficientNet with basic noise augmentation using real audio recordings mixed with synthetic drums. Implements standard noise mixing without acoustic matching, focusing on general noise robustness. Includes checkpoint resumption support and comprehensive evaluation on both clean and noisy test sets.

- **`05_train_effnet_ploss_randnoise.py`** - Evaluation-focused script (training disabled by default) that tests models trained with random noise augmentation. Implements `noise_mode="random"` and provides comparative evaluation between synthetic-only and noise-augmented datasets. Designed for systematic robustness assessment.

- **`05_train_effnet_ploss_randpratm.py`** - Trains EfficientNet with random PRATM (Physical Real-Audio Transient Modeling) noise augmentation. Uses `noisemodel="pratm"` for physics-informed noise modeling and provides dual evaluation on synthetic and real-audio test sets. Includes comprehensive model save path configuration with noise model specification.

### Advanced Noise Modeling
- **`06_train_effnet_ploss_gaussnoise.py`** - Evaluation script for Gaussian noise robustness testing with `noise_mode="gaussian"` and `noisemodel="pratm"`. Focuses on synthetic data evaluation while commenting out noise evaluation sections. Uses physics-informed transient modeling for controlled noise experiments.

- **`07_train_effnet_ploss_statgaussnoise.py`** - Trains EfficientNet with stationary Gaussian noise augmentation using `noise_mode="statgauss"`. Designed for controlled noise robustness studies with predictable noise characteristics. Focuses on noise-augmented evaluation while commenting out clean synthetic testing.

- **`10_train_effnet_ploss_mixnoise.py`** - Trains EfficientNet with mixed noise strategies combining multiple noise models using `noise_mode="mix"`. Uses `noisemodel="pratm"` with randomized two-component mixing for diverse acoustic conditions. Provides synthetic-only evaluation while commenting out noise evaluation.

- **`11_train_effnet_ploss_randnoise.py`** - Full training and evaluation pipeline for random noise augmentation with `noisemodel="noise"`. Implements comprehensive dual evaluation on both synthetic and noise-augmented test sets for complete robustness assessment.

- **`12_train_effnet_ploss_randtransient.py`** - Trains EfficientNet with random transient noise modeling using `noisemodel="transient"`. Focuses on transient-specific noise characteristics for drum synthesis applications. Includes full dual evaluation pipeline for comprehensive noise robustness testing.

## Diffusion Models for Noise Generation

### Diffusion Training and Evaluation
- **`08_train_diffuser.py`** - Trains a 1D UNet diffusion model for noise generation using the DanceDiffusion pipeline. Implements custom `NoiseData` dataset class for loading real audio with onset detection and temporal alignment. Uses SophiaG optimizer with cosine learning rate scheduling and mixed precision training (fp16). Designed for generating realistic acoustic noise for data augmentation with 64-step denoising process.

- **`09_eval_diffuser.py`** - Evaluation script for pre-trained diffusion models that generates noise samples from trained checkpoints. Creates batch audio generation with reproducible seeding and saves generated noise samples to organized output directories. Designed for systematic evaluation of diffusion-generated noise quality.

## Data Structure

The `data/` directory contains:
- **`full_param_log.csv`** - Complete perceptual parameter dataset (5-6 parameters) with train/val/test fold assignments
- **`full_param_log_phys.csv`** - Physics-based parameter dataset with physical modeling parameters
- **`full_param_log_phys_filtered.csv`** - Filtered version of physics parameters for improved synthesis stability
- HDF5 files generated by pipeline scripts for efficient audio and parameter storage

## Key Features

### Noise Robustness Focus
- **Multiple Noise Strategies**: Random, matched, Gaussian, stationary, mixed, and transient noise augmentation
- **Physics-Informed Noise**: PRATM modeling for realistic drum/percussion noise characteristics
- **Diffusion-Generated Noise**: Advanced generative modeling for creating diverse acoustic conditions
- **Systematic Evaluation**: Comprehensive testing on both clean synthetic and noise-augmented datasets

### Advanced Training Infrastructure
- **Multi-Platform Support**: CUDA GPU training with comprehensive device detection
- **Flexible Parameter Scaling**: MinMax normalization with log-scale parameter handling
- **Checkpoint Management**: Dual checkpoint saving (best validation and latest) with resumption support
- **Comprehensive Logging**: TensorBoard integration with detailed training metrics

### Synthesis Parameters
- **Physics Mode**: 6 parameters (EI, Ts0, d1, d3, lm, ell) for physical modeling
- **Perceptual Mode**: 5-6 parameters (w1, tau, p, D, lm, ell) for perceptual matching
- **Parameter Scaling**: Log-scale transformation for frequency parameters, MinMax normalization to [-1, 1]
- **Position Randomization**: Randomized excitation positions for increased synthesis diversity

## Usage Notes

All training scripts accept command-line arguments for flexible configuration:
- `save_dir`: Data directory path
- `init_id`: Experiment initialization ID
- `minmax`: MinMax scaling flag (0/1)
- `logscale_theta`: Log-scale parameter flag (0/1)
- `opt`: Optimizer choice (Adam, Sophia)
- `synth_type`: Synthesis type (ftm, amchirp, string)
- `ckpt_path`: Checkpoint path for resumption

The experiments focus on understanding how different noise augmentation strategies affect neural parameter estimation robustness, with particular emphasis on real-world drum recording analysis. The pipeline follows: data generation → training with noise augmentation → evaluation on clean and noisy test sets → robustness analysis.
