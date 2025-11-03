# Adam vs Sophia Optimizer Comparison

## Overview

This guide helps you compare **Adam** (first-order) vs **Sophia** (second-order) optimizers for percussion synthesis parameter estimation.

## What is Sophia?

**Sophia (Second-order Clipped Stochastic Optimization)** is a modern optimizer that uses Hessian (curvature) information for more efficient training.

### Key Differences

| Feature | Adam | Sophia |
|---------|------|--------|
| **Order** | First-order (gradients only) | Second-order (gradients + Hessian) |
| **Memory** | Lower (2 states per parameter) | Higher (3 states per parameter) |
| **Computation** | Faster per step | Slower (Hessian updates) |
| **Convergence** | Good, well-tested | Potentially better with curvature info |
| **Stability** | Stable with proper LR | Clipped Hessian prevents instability |

### Mathematical Updates

**Adam:**
```
m_t = β₁·m_{t-1} + (1-β₁)·∇L      # First moment (mean)
v_t = β₂·v_{t-1} + (1-β₂)·∇L²     # Second moment (variance)
θ = θ - lr · m_t / (√v_t + ε)
```

**Sophia:**
```
m_t = β₁·m_{t-1} + (1-β₁)·∇L      # First moment (mean)
h_t = β₂·h_{t-1} + (1-β₂)·∇²L     # Hessian diagonal
θ = θ - lr · m_t / max(ρ·h_t, ε)  # Clipped by curvature
```

Where:
- `ρ` (rho) = 0.01: Hessian clipping parameter
- `β₁` = 0.965, `β₂` = 0.99: Momentum parameters
- `weight_decay` = 0.1: Regularization

## Running the Comparison

### Quick Start

```bash
cd /Users/jos/w/perceptual_neural_physical

# Run comparison with default settings (10 epochs, quick test)
./icassp25/compare_optimizers.sh outputs/icassp25 test_001
```

### Arguments

```bash
./icassp25/compare_optimizers.sh <save_dir> <init_id>
```

- `save_dir`: Output directory (default: `outputs/icassp25`)
- `init_id`: Experiment identifier (default: `test`)

### What It Does

1. Trains **EfficientNet-B0** with **Adam** optimizer
2. Trains **EfficientNet-B0** with **Sophia** optimizer
3. Both use:
   - P-Loss (basic parameter MSE)
   - MinMax scaling [-1, 1]
   - Log-scale for frequency parameters
   - Batch size: 32
   - Learning rate: 0.001
   - 10 epochs (rapid prototyping)

### Output Structure

```
outputs/icassp25/f_W/
├── b0_ploss_finetuneFalse_log-1_minmax-1_opt-adam_batch_size32_lr-0.001_init-test_001_adam/
│   ├── logs/                  # TensorBoard logs
│   ├── *.ckpt                 # Model checkpoints
│   └── test_predictions.npy   # Final predictions
└── b0_ploss_finetuneFalse_log-1_minmax-1_opt-sophia_batch_size32_lr-0.001_init-test_001_sophia/
    ├── logs/                  # TensorBoard logs
    ├── *.ckpt                 # Model checkpoints
    └── test_predictions.npy   # Final predictions
```

## Analyzing Results

### 1. TensorBoard Comparison

View training curves side-by-side:

```bash
tensorboard --logdir=outputs/icassp25/f_W/
```

Open browser to `http://localhost:6006`

**Key metrics to compare:**
- `train_loss` vs `val_loss` - Convergence speed
- `lr` - Learning rate schedule
- Gradient norms - Optimization stability

### 2. Best Checkpoint Analysis

Find the best performing checkpoint:

```bash
# For Adam
python print_best_checkpoint.py outputs/icassp25/f_W/b0_ploss_*_adam/

# For Sophia
python print_best_checkpoint.py outputs/icassp25/f_W/b0_ploss_*_sophia/
```

This will show:
- Best validation loss
- Epoch number
- Checkpoint filename

### 3. Test Set Evaluation

Load checkpoints and evaluate on test set:

```bash
python icassp25/01_eval_ploss.py outputs/icassp25 test_001_adam
python icassp25/01_eval_ploss.py outputs/icassp25 test_001_sophia
```

### 4. Audio Comparison

Generate and compare synthesized audio:

```bash
# Generate audio comparisons
python icassp25/audio_comparison.py \
    outputs/icassp25/f_W/b0_ploss_*_adam/step=*bestckpt*.ckpt \
    outputs/icassp25

# Create interactive HTML comparison
python icassp25/generate_comparison_html.py \
    outputs/icassp25/audio_comparisons/
```

## Expected Behavior

### Adam (Baseline)
- **Pros**: Fast, stable, well-understood
- **Cons**: May converge slower in complex loss landscapes
- **Typical val_loss after 10 epochs**: ~0.05-0.10 (depends on initialization)

### Sophia (Second-Order)
- **Pros**: Better convergence in theory, uses curvature information
- **Cons**: Slightly slower per epoch, more memory
- **Typical val_loss after 10 epochs**: Similar or better than Adam

### What to Look For

**Sophia wins if:**
- Lower validation loss after same epochs
- Faster convergence (reaches good loss in fewer epochs)
- More stable training (smoother loss curves)

**Adam wins if:**
- Similar final loss with faster wall-clock time
- More stable across different initializations
- Lower memory usage matters

## Troubleshooting

### Sophia runs slower
- **Expected**: Hessian computation adds ~10-20% overhead
- **Solution**: If wall-clock time matters more than convergence, use Adam

### Out of memory errors
- **Cause**: Sophia stores additional Hessian states
- **Solution**: Reduce batch size or use Adam

### Sophia diverges
- **Rare but possible**: Adjust `rho` parameter in `src/pnp_synth/neural/cnn.py:380`
- **Default**: `rho=0.01` (works for most cases)
- **Try**: `rho=0.04` for more aggressive clipping

## Advanced: Running Full Experiments

For publication-quality results (70 epochs):

```bash
# Modify epoch_max in the training script
# Edit icassp25/03_train_effnet_ploss.py line 52:
# epoch_max = 70  # Change from 10 to 70

# Then run comparison
./icassp25/compare_optimizers.sh outputs/icassp25 full_001
```

## References

- **Sophia Paper**: [Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)
- **Implementation**: `src/pnp_synth/neural/optimizer.py` (SophiaG class)
- **Usage**: `src/pnp_synth/neural/cnn.py:379-391` (EffNet.configure_optimizers)

## Summary

**Quick Recommendation:**
1. **Start with Adam** - proven, fast, stable baseline
2. **Try Sophia** if you want potentially better convergence
3. **Compare both** using TensorBoard and validation loss
4. **Pick the winner** based on your priorities (speed vs convergence)

For percussion synthesis parameter estimation, both should work well. The best choice depends on your specific dataset and computational budget.
