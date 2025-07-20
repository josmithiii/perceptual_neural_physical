# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for Perceptual-Neural-Physical (PNP) sound matching, focusing on neural parameter estimation for physics-based audio synthesis. The project bridges physical modeling, perceptual analysis, and deep learning for musical timbre modeling.

## Development Setup

### Environment Setup
```bash
# Create and activate uv environment
./setup.sh

# Manual setup if needed:
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install torch torchvision torchaudio

# Install kymatio (JTFS-GPU version)
git clone https://github.com/cyrusvahidi/jtfs-gpu.git
cd jtfs-gpu
uv pip install -e .
cd ..
```

### Running Experiments
```bash
# Navigate to specific experiment directory
cd icassp23/  # or gretsi23/, taslp23/, etc.

# Run numbered pipeline scripts in order:
python 01_generate_audio.py      # Generate synthetic training data
python 02_compute_pnp_jacobian.py  # Compute gradients for PNP loss
python 03_train_effnet_ploss.py  # Train with basic parameter loss
python 05_train_effnet_pnploss.py  # Train with PNP loss

# For cluster execution:
cd sbatch/
sbatch 03_train_effnet_ploss.py  # Submit to SLURM
```

## Core Architecture

### Three-Component Pipeline (PNP)
1. **Physical (`src/pnp_synth/physical/`)**: Differentiable synthesis models
   - `ftm.py`: Functional Transformation Method for drums (PDE-based)
   - `amchirp.py`: Amplitude-modulated chirp synthesis
   
2. **Neural (`src/pnp_synth/neural/`)**: Parameter estimation networks
   - `cnn.py`: EfficientNet-based architecture
   - `loss.py`: Specialized loss functions (P-loss, Spec-loss, PNP-loss)
   - `optimizer.py`: Custom optimizers including Levenberg-Marquardt variants

3. **Perceptual (`src/pnp_synth/perceptual/`)**: Perceptual similarity metrics
   - `jtfs.py`: Joint Time-Frequency Scattering parameters
   - `metrics.py`: Multi-scale perceptual evaluation

### Key Innovation: PNP Loss Function
The core contribution is the PNP loss that uses gradients from the physical synthesis model:
```python
# Compute gradients: ∇_θ (Φ ∘ g)(θ) where Φ=JTFS, g=synthesizer
J = dS_over_dnu(theta)  # Jacobian computation
M = torch.mm(J.T, J)    # Riemannian metric
# Use M for weighted parameter loss
```

## Experiment Organization

### Conference-Based Structure
- `icassp23/`, `gretsi23/`, `taslp23/`: Published work
- `icassp25/`, `mersenne24/`: Current research
- Each contains complete pipeline from data generation to evaluation

### Numbered Pipeline Pattern
1. `01_generate_audio.py`: Create synthetic data using physical models
2. `02_compute_pnp_jacobian.py`: Pre-compute gradients for efficiency
3. `03-07_train_effnet_*.py`: Various training configurations
4. `0X_eval_*.py`: Evaluation and analysis scripts

### Data Management
- `data/full_param_log.csv`: Complete parameter dataset
- `data/{train,val,test}_param_log.csv`: Standard splits
- HDF5 files for efficient audio and gradient storage

## Parameter Scaling Strategy
- **Log-scale transformation** for frequency parameters (omega, p, D)
- **MinMax scaling** to [-1, 1] for neural network compatibility
- **Differentiable inverse scaling** in synthesis forward pass

## Loss Function Hierarchy
- **P-Loss**: Basic parameter MSE (baseline)
- **Spec-Loss**: Multi-scale spectral reconstruction loss
- **PNP-Loss**: Perceptual-Neural-Physical loss using computed gradients
- **Adaptive variants**: Levenberg-Marquardt-style weight scheduling

## Dependencies
- **kymatio**: JTFS implementation (use cyrusvahidi/jtfs-gpu fork)
- **pytorch-lightning**: Training framework
- **functorch**: Auto-differentiation for Jacobian computation
- **auraloss**: Perceptual audio loss functions
- **librosa, soundfile**: Audio processing
- **h5py**: Efficient data storage

## Development Notes
- Each experiment directory has a main config file (e.g., `icassp23.py`)
- Use `sbatch/` scripts for cluster computation
- Physical synthesis models are differentiable end-to-end
- Parameter ranges and scaling are critical for convergence