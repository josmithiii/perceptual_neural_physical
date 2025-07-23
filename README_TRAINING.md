# EfficientNet Training Guide for FTM Dataset with Jacobians

This document provides best practices for training EfficientNet models
on the Functional Transformation Method (FTM) dataset using
pre-computed Jacobians for physics-informed learning.

[Written by Claude Sonnet 4 Code based on its reading of the files herein.]

## Quick Start

### 1. Pre-compute Jacobians
```bash
cd taslp23/
python 02_compute_pnp_jacobian.py <save_dir> <id_start> <id_end> <logscale> 0 <minmax>
```

### 2. Train EfficientNet with PNP Loss
```bash
python 05_train_effnet_pnploss.py <save_dir> <init_id> <minmax> <logscale_theta> <optimizer> ftm
```

## Optimal Training Configuration

### Model Architecture
- **Network**: EfficientNet-B0 (`cnn_type = "efficientnet"`)
- **Input channels**: 1 (CQT spectrogram)
- **Output dimension**: 5 (FTM parameters: ω, τ, p, D, α)
- **Activation**: Tanh (for MinMax scaled parameters)

### Training Hyperparameters
- **Batch size**: 256 (for PNP loss training)
- **Epochs**: 70
- **Learning rate**: 1e-3
- **Optimizer**: Adam
- **Loss type**: `"weighted_p"` (PNP loss with Jacobian weighting)
- **Weight type**: `"novol"` (no Riemannian volume weights)

### FTM-Specific Parameters
- **Sample rate**: 22,050 Hz
- **Scattering scale (J)**: 10
- **Filters per octave (Q)**: 12
- **BatchNorm variance**: 0.5

## Levenberg-Marquardt Algorithm (LMA) Configuration

**Recommended settings for optimal convergence:**
```python
LMA = {
    'mode': "adaptive",     # Adaptive damping strategy
    'accelerator': 0.05,    # Lambda reduction factor (good performance)
    'brake': 1,             # Lambda increase factor (conservative)
    'damping': "id"         # Identity matrix damping
}
```

- **mu**: 1e-10 (scaling factor for Riemannian metric M)

## Parameter Scaling Strategy

### FTM Physical Parameters
1. **ω (omega)**: Frequency parameter → log-scaled
2. **τ (tau)**: Decay parameter → linear
3. **p**: Frequency parameter → log-scaled
4. **D**: Diffusion parameter → log-scaled
5. **α (alpha)**: Amplitude parameter → linear

### Scaling Pipeline
1. **Log-scale transformation**: Applied to frequency parameters (ω, p, D)
2. **MinMax normalization**: Scale to [-1, 1] for neural network compatibility
3. **Differentiable inverse scaling**: Applied in synthesis forward pass

## Jacobian Computation Details

### Forward Operator Chain
```
Parameters (nu) → Scaled Parameters (theta) → Audio (x) → JTFS Coefficients (S)
```

### Jacobian Calculation
- **Method**: Forward-mode autodiff (`jacfwd`) - efficient for low-dimensional input, high-dimensional output
- **Computation**: `dS_over_dnu = jacfwd(S_from_nu)`
- **Storage**: Saved as `.npy` files for efficient batch loading during training

### JTFS Configuration
```python
jtfs_params = {
    'J': 13,              # Scattering scale ~1000ms
    'shape': (2**16,),    # Input duration ~3 seconds
    'Q': (12, 1),         # Filters per octave (1st, 2nd order)
    'Q_fr': 1,            # Frequency filters per octave
    'F': 2,               # Local frequency averaging
    'max_pad_factor': 1,   # Temporal padding limit
    'max_pad_factor_fr': 1, # Frequency padding limit
    'pad_mode': 'zero',
    'pad_mode_fr': 'zero'
}
```

## Loss Function Details

### PNP Loss Components
1. **Physical**: Differentiable synthesis model gradients
2. **Neural**: EfficientNet parameter estimation
3. **Perceptual**: JTFS-based similarity metric

### Riemannian Metric Weighting
- **Metric matrix**: `M = J^T * J` (where J is the Jacobian)
- **Weighted loss**: Uses M for physics-informed parameter importance
- **Volume weighting**: Disabled (`weight_type = "novol"`) for optimal performance

## Training Performance Tips

### Device Optimization
- **CUDA**: Full GPU acceleration with gradient analysis
- **MPS (Apple Silicon)**: Use float32 precision
- **CPU**: Fallback with optimized threading

### Memory Management
- **HDF5 storage**: Efficient loading of audio and Jacobian matrices
- **Batch processing**: 256 samples per batch for optimal GPU utilization
- **Data workers**: Set `num_workers=0` to avoid multiprocessing issues

## Expected Results

### Training Metrics
- **Convergence**: Typically within 70 epochs
- **Validation loss**: Monitor for early stopping
- **Test performance**: Evaluated on held-out FTM parameter sets

### Model Outputs
- **Checkpoints**: Saved to `<save_dir>/f_W/` with experiment configuration
- **Predictions**: Test set predictions saved as `test_predictions.npy`
- **Logs**: TensorBoard logs for training visualization

## File Structure

```
<save_dir>/
├── x/                    # Audio data (HDF5 files)
├── M/                    # Jacobian matrices
├── f_W/                  # Trained models
└── logs/                 # Training logs
```

## Reference Implementation

The authoritative implementation is found in:
- **Training script**: `taslp23/05_train_effnet_pnploss.py`
- **Jacobian computation**: `taslp23/02_compute_pnp_jacobian.py`
- **Model architecture**: `src/pnp_synth/neural/cnn.py`

This configuration represents the state-of-the-art approach validated through the TASLP23 publication.
