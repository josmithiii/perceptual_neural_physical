# Audio Comparison System: Target vs Reconstruction

This system allows you to audition and compare drum synthesis results between ground truth parameters (target) and model predictions (reconstruction).

## Quick Start

After your model evaluation is complete:

```bash
# Navigate to icassp25 directory
cd icassp25/

# Run complete comparison workflow (generates 10 samples by default)
./compare_audio.sh outputs/icassp25/f_W/b0_ploss_finetuneFalse_log-1_minmax-1_opt-adam_batch_size32_lr-0.001_init-test

# Or specify custom number of samples
./compare_audio.sh <model_dir> 20 ./my_comparison
```

## What You Get

### 🎵 Audio Files
- `sample_001_target.wav` - Original synthesis from ground truth parameters
- `sample_001_reconstruction.wav` - Synthesis from model predictions
- Pairs for each test sample

### 📊 Analysis Data
- `analysis.json` - Comprehensive parameter errors and sample metadata
- Parameter prediction errors (MAE, MSE, RMSE) for each dimension
- Sample-by-sample error breakdown

### 🌐 Interactive Interface
- `comparison.html` - Side-by-side audio comparison interface
- Navigate between samples with arrow keys or buttons
- View target vs predicted parameters with error highlighting
- Audio players for immediate A/B comparison

## Individual Scripts

### 1. Generate Audio Comparisons
```bash
python audio_comparison.py <model_dir> --num_samples=10 --output_dir=./audio_comparison
```

**Parameters:**
- `model_dir`: Path to trained model directory containing `test_predictions_epoch*.npy`
- `--num_samples`: Number of test samples to process (default: 10)
- `--output_dir`: Output directory for audio files and analysis

### 2. Create HTML Interface
```bash
python generate_comparison_html.py analysis.json --output=comparison.html
```

## Model Directory Structure

The system expects your model directory to contain:
```
model_dir/
├── test_predictions_epoch07.npy    # Model predictions (latest epoch used)
├── test_predictions_epoch10.npy
└── ... (other model files)
```

## Parameter Analysis

The system analyzes prediction accuracy for all FTM drum parameters:
- **omega**: Fundamental frequency
- **tau**: Decay time constant
- **p**: Resonance parameter
- **D**: Damping coefficient
- **alpha**: Amplitude scaling

## HTML Interface Features

- **Audio Players**: Side-by-side target vs reconstruction comparison
- **Parameter Display**: Ground truth vs predicted values with error highlighting
- **Navigation**: Arrow keys, buttons, or dropdown selection
- **Error Coding**: Color-coded parameter errors (green=good, yellow=medium, red=high)
- **Summary Statistics**: Overall model performance metrics

## Tips for Analysis

1. **Listen for timbral differences** - How well does the model capture drum character?
2. **Check parameter errors** - Which parameters are hardest to predict?
3. **Identify failure cases** - Samples with high reconstruction error
4. **Correlate metrics** - How do parameter errors relate to perceptual quality?

## Integration with Evaluation

The audio comparison uses the same synthesis pipeline as your PNP metrics, ensuring consistency between quantitative evaluation and qualitative audition.

This bridges the gap between numbers (JTFS metrics, MSS loss) and actual sound quality! 🎯
