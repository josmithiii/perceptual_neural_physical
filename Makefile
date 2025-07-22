# Makefile for Perceptual Neural Physical Sound Matching
# Research codebase with experiment pipelines and development workflows

SHELL := /bin/bash

.PHONY: help setup install test clean lint format jupyter experiments

# Default target - FIXME: NEEDS UPDATING
h help:
	@echo "Perceptual Neural Physical Sound Matching - Development Makefile"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup          - Full environment setup using uv"
	@echo "  install        - Install package in editable mode"
	@echo "  clean          - Remove build artifacts and cache files"
	@echo ""
	@echo "Testing and Validation:"
	@echo "  test           - Run all import and functionality tests"
	@echo "  test-import    - Test package imports"
	@echo "  test-modules   - Test individual module imports"
	@echo "  test-synthesis - Test physical synthesis modules"
	@echo "  test-neural    - Test neural network modules"
	@echo "  test-torch     - Test PyTorch and CUDA availability"
	@echo ""
	@echo "Development:"
	@echo "  jupyter        - Start Jupyter Lab"
	@echo "  lint           - Run code linting (if available)"
	@echo "  format         - Format code (if available)"
	@echo ""
	@echo "Data Generation:"
	@echo "  ai25           - Generate ICASSP25 audio data (full dataset)"
	@echo "  ai25test       - Generate ICASSP25 test audio (100 samples per fold)"
	@echo "  Mi25           - Compute ICASSP25 PNP Jacobian matrices"
	@echo "  asm24          - Generate MERSENNE24 string perceptual audio data"
	@echo "  asm24phys      - Generate MERSENNE24 string physics-based audio data"
	@echo "  asm24all       - Generate all MERSENNE24 string audio data"
	@echo "  csm24          - Clean MERSENNE24 string synthesis audio data"
	@echo "  at23           - Generate TASLP23 audio data"
	@echo "  jt23           - Compute TASLP23 Jacobians"  
	@echo "  Mt23           - Compute TASLP23 LMA matrices"
	@echo "  mht23          - Merge TASLP23 H5 files"
	@echo "  cat23          - Clean TASLP23 audio data"
	@echo ""
	@echo "Training:"
	@echo "  ri25pl         - Train EfficientNet with P-loss (ICASSP25)"
	@echo "  ri25pnpl       - Train EfficientNet with PNP-loss (ICASSP25)"
	@echo "  ri25all        - Run all ICASSP25 training variants"
	@echo "  rm24pl         - Train EfficientNet with P-loss (MERSENNE24 string synthesis)"
	@echo "  rm24matched    - Train with matched noise augmentation (MERSENNE24 strings)"
	@echo "  rm24noise      - Train with basic noise augmentation (MERSENNE24 strings)"
	@echo "  rm24pratm      - Train with PRATM noise augmentation (MERSENNE24 strings)"
	@echo "  rm24gauss      - Train with Gaussian noise augmentation (MERSENNE24 strings)"
	@echo "  rm24transient  - Train with transient noise augmentation (MERSENNE24 strings)"
	@echo "  rm24diffuse    - Train diffusion model for noise generation (MERSENNE24)"
	@echo "  rm24noise-all  - Run all MERSENNE24 string noise robustness training"
	@echo "  rm24all        - Run complete MERSENNE24 string training pipeline"
	@echo "  ri23pl         - Train EfficientNet with P-loss (ICASSP23)"
	@echo "  ri23pnpl       - Train EfficientNet with PNP-loss (ICASSP23)"
	@echo "  rt23pl         - Train EfficientNet with P-loss (TASLP23)"
	@echo "  rt23sl         - Train EfficientNet with Spectral-loss (TASLP23)"
	@echo "  rt23pnpl       - Train EfficientNet with PNP-loss (TASLP23)"
	@echo "  rt23ft-pnpl    - Fine-tune with PNP-loss (TASLP23)"
	@echo "  rt23ft-mss     - Fine-tune with MSS-loss (TASLP23)"
	@echo "  rt23all        - Run all TASLP23 training variants"
	@echo ""
	@echo "Evaluation:"
	@echo "  ei25pl         - Evaluate P-loss model (ICASSP25)"
	@echo "  ei25pnp        - Evaluate PNP model (ICASSP25)"
	@echo "  ei25grad       - Evaluate gradients (ICASSP25, batch_size=32)"
	@echo "  ei25grad_ft    - Evaluate gradients for fine-tuning (ICASSP25)"
	@echo "  ei25grad_original - Evaluate gradients (ICASSP25, batch_size=256)"
	@echo "  ei25audio      - Generate audio comparisons (ICASSP25)"
	@echo "  ei25html       - Generate HTML comparison interface (ICASSP25)"
	@echo "  em24diffuse    - Evaluate diffusion model and generate samples (MERSENNE24)"
	@echo "  em24eval       - Evaluate MERSENNE24 models (placeholder)"
	@echo ""
	@echo "Variables:"
	@echo "  SAVE_DIR       - Output directory for experiments (default: ./outputs)"
	@echo "  INIT_ID        - Experiment initialization ID (default: test)"
	@echo "  BATCH_SIZE     - Training batch size (default: 32)"

# Variables
SAVE_DIR ?= ./outputs
INIT_ID ?= test
BATCH_SIZE ?= 32
VENV_PYTHON = .venv/bin/python

# Setup and Installation
setup:
	@echo "Setting up development environment..."
	./setup.sh

install:
	@echo "Installing package in editable mode..."
	source .venv/bin/activate && uv pip install -e .


#================================== USAGE MAKE TARGETS =====================================

# For data generation, training, and model evaluation, organized by conference/journal-article project

# -------------------- ICASSP-25  -------------------

# File target - only runs if output doesn't exist
$(SAVE_DIR)/icassp25/x/ftm_train_audio.h5:
	@mkdir -p $(SAVE_DIR)/icassp25
	source .venv/bin/activate && python icassp25/01_generate_audio.py $(SAVE_DIR)/icassp25

# ICASSP25 data generation using FTM modal drum synthesis
ai25 audio-icassp25: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5
	@echo "✓ ICASSP25 audio data generated"

# Test audio generation (first 100 samples per fold)
ai25test audio-icassp25-test:
	@echo "Generating ICASSP25 test audio data (100 samples per fold)..."
	@mkdir -p $(SAVE_DIR)/icassp25
	source .venv/bin/activate && python icassp25/01_generate_audio_test.py $(SAVE_DIR)/icassp25
	@echo "✓ ICASSP25 test audio data generated"

# Compute PNP Jacobian matrices
Mi25 M-icassp25: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5
	@echo "Computing PNP Jacobian matrices for ICASSP25..."
	source .venv/bin/activate && python icassp25/01b_compute_pnp_jacobian.py $(SAVE_DIR)/icassp25

# Training targets
ri25pl run-icassp25-ploss: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5
	@echo "Training EfficientNet with P-loss (ICASSP25)..."
	source .venv/bin/activate && python icassp25/03_train_effnet_ploss.py $(SAVE_DIR)/icassp25 $(INIT_ID) 1 1 adam b0 $(BATCH_SIZE)

ri25pnpl run-icassp25-pnploss: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5 Mi25
	@echo "Training EfficientNet with PNP-loss (ICASSP25)..."
	source .venv/bin/activate && python icassp25/05_train_effnet_pnploss.py $(SAVE_DIR)/icassp25 $(INIT_ID) 1 1 adam b0 $(BATCH_SIZE)

ri25all run-icassp25-all: ri25pl ri25pnpl

# Evaluation targets
ei25pl eval-icassp25-ploss:
	@echo "Evaluating P-loss model (ICASSP25)..."
	source .venv/bin/activate && python icassp25/01_eval_ploss.py b0 adam $(INIT_ID) > ei25pl_log_$$(date +%Y-%m-%d).txt 2>&1 &
	@echo "Background evaluation started. Check log: icassp25/ei25pl_log_$$(date +%Y-%m-%d).txt"

ei25pnp eval-icassp25-pnp:
	@echo "Evaluating PNP model (ICASSP25)..."
	source .venv/bin/activate && python icassp25/02_eval_pnp.py b0 adam 0

# Gradient analysis targets
ei25grad eval-icassp25-grad:
	@echo "Evaluating gradients (ICASSP25)..."
	source .venv/bin/activate && python icassp25/06_eval_grad.py adam ploss b0 32

ei25grad_ft eval-icassp25-grad-finetune:
	@echo "Evaluating gradients for fine-tuning (ICASSP25)..."
	source .venv/bin/activate && python icassp25/08_eval_grad_finetune.py adam

ei25grad_original eval-icassp25-grad-original:
	@echo "Evaluating gradients (ICASSP25) - original config..."
	source .venv/bin/activate && python icassp25/06_eval_grad.py adam ploss b0 256

# Audio comparison and analysis
ei25a eval-icassp25-audio-comparison:
	@echo "Generating audio comparison (ICASSP25)..."
	source .venv/bin/activate && python icassp25/audio_comparison.py $(SAVE_DIR)/icassp25/f_W/b0_ploss_finetuneFalse_log-1_minmax-1_opt-adam_batch_size$(BATCH_SIZE)_lr-0.001_init-$(INIT_ID) --num_samples=10 --output_dir=./icassp25/audio_comparison

ei25html eval-icassp25-html:
	@echo "Generating HTML comparison interface (ICASSP25)..."
	source .venv/bin/activate && python icassp25/generate_comparison_html.py ./icassp25/audio_comparison/analysis.json --output=./icassp25/audio_comparison/comparison.html
	@echo "✓ Interactive HTML comparison generated at icassp25/audio_comparison/comparison.html"

# -------------------- MERSENNE-24  -------------------

# File targets - only run if outputs don't exist
$(SAVE_DIR)/mersenne24/x/mersenne24_train_audio.h5:
	@mkdir -p $(SAVE_DIR)/mersenne24
	source .venv/bin/activate && python mersenne24/01_generate_audio.py $(SAVE_DIR)/mersenne24

$(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5:
	@mkdir -p $(SAVE_DIR)/mersenne24
	source .venv/bin/activate && python mersenne24/01_generate_audio_phys.py $(SAVE_DIR)/mersenne24

# Primary string data generation targets
asm24 audio-strings-mersenne24: $(SAVE_DIR)/mersenne24/x/mersenne24_train_audio.h5
	@echo "✓ MERSENNE24 string perceptual audio data generated (6 params: w1, tau, p, D, lm, ell)"

asm24phys audio-strings-mersenne24-physics: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "✓ MERSENNE24 string physics-based audio data generated (6 params: EI, Ts0, d1, d3, lm, ell)"

asm24all audio-strings-mersenne24-all: asm24 asm24phys
	@echo "✓ All MERSENNE24 string synthesis audio data generated"

# Training targets - Baseline (using string synthesis)
rm24pl run-mersenne24-ploss: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with P-loss (MERSENNE24 string synthesis)..."
	source .venv/bin/activate && python mersenne24/03_train_effnet_ploss.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

# Training targets - Noise robustness experiments
rm24matched run-mersenne24-matched-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with matched noise augmentation (MERSENNE24 string synthesis)..."
	source .venv/bin/activate && python mersenne24/04_train_effnet_ploss_matchednoise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24noise run-mersenne24-basic-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with basic noise augmentation (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/04_train_effnet_ploss_noise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24randnoise run-mersenne24-random-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with random noise augmentation (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/05_train_effnet_ploss_randnoise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24pratm run-mersenne24-pratm-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with PRATM noise augmentation (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/05_train_effnet_ploss_randpratm.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24gauss run-mersenne24-gaussian-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with Gaussian noise augmentation (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/06_train_effnet_ploss_gaussnoise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24statgauss run-mersenne24-stationary-gaussian: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with stationary Gaussian noise (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/07_train_effnet_ploss_statgaussnoise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24mix run-mersenne24-mixed-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with mixed noise strategies (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/10_train_effnet_ploss_mixnoise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24randnoise2 run-mersenne24-random-noise2: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with random noise v2 (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/11_train_effnet_ploss_randnoise.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

rm24transient run-mersenne24-transient-noise: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training EfficientNet with transient noise augmentation (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/12_train_effnet_ploss_randtransient.py $(SAVE_DIR)/mersenne24 $(INIT_ID) 1 1 adam string None

# Diffusion model targets
rm24diffuse train-mersenne24-diffuser: $(SAVE_DIR)/mersenne24/x/mersenne24_phys_train_audio.h5
	@echo "Training diffusion model for noise generation (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/08_train_diffuser.py

em24diffuse eval-mersenne24-diffuser:
	@echo "Evaluating diffusion model and generating samples (MERSENNE24)..."
	source .venv/bin/activate && python mersenne24/09_eval_diffuser.py

# Comprehensive training pipelines
rm24noise-all run-mersenne24-noise-all: rm24matched rm24noise rm24randnoise rm24pratm rm24gauss rm24statgauss rm24mix rm24randnoise2 rm24transient
	@echo "✓ All MERSENNE24 string synthesis noise robustness training completed"

rm24all run-mersenne24-all: rm24pl rm24noise-all rm24diffuse
	@echo "✓ Complete MERSENNE24 string synthesis training pipeline finished"

# Evaluation targets (placeholder - scripts not yet implemented)
em24eval eval-mersenne24-models:
	@echo "MERSENNE24 evaluation targets - to be implemented based on specific evaluation needs"
	@echo "Available trained models in: $(SAVE_DIR)/mersenne24/f_W/"

# Clean targets
cm24 clean-mersenne24:
	-/bin/rm -rf $(SAVE_DIR)/mersenne24/x/
	@echo "✓ MERSENNE24 string synthesis audio data cleaned"

csm24 clean-strings-mersenne24: cm24
	@echo "✓ MERSENNE24 string synthesis audio data cleaned (alias for cm24)"

# ---------------------- TASLP-23  --------------------

# Based on commit history, the ./taslp23/ fine-tuned PNP subproject is
# most likely the "Current" project demonstrated at
# https://lylyhan.github.io/perceptual_neural_physical/ They really
# sound best, so it matters!  See README_TIMELINE.md for Claude's
# analysis.  According to Claude, the git commits show active TASLP23
# development continuing into 2025 (commits like "mersenne data v3",
# "update dataset", etc.), suggesting TASLP23 represents the
# current/latest research direction at the time the website examples
# were generated.

# File target - only runs if output doesn't exist  
$(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	source .venv/bin/activate && python taslp23/01_generate_audio.py $(SAVE_DIR)/taslp23

at23 audio-taslp23: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "✓ TASLP23 audio data generated"

cat23 clean-audio-taslp23:
	-/bin/rm -rf $(SAVE_DIR)/taslp23/x/

# Compute PNP Jacobians for TASLP23 to ./taslp23/outputs/taslp23/x/
jt23 jacobian-taslp23: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Computing PNP Jacobian matrices for TASLP23..."
	source .venv/bin/activate && python taslp23/02_compute_pnp_jacobian.py $(SAVE_DIR)/taslp23 0 1000 1 0 1

# Compute LMA step matrices
Mt23 M-taslp23: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Computing LMA step matrices for TASLP23..."
	source .venv/bin/activate && python taslp23/12_compute_lmastep.py $(SAVE_DIR)/taslp23 0 1000

# Merge H5 files
mht23 merge-h5-taslp23:
	@echo "Merging H5 files for TASLP23..."
	source .venv/bin/activate && python taslp23/13_merge_h5file.py

# Basic training targets
rt23pl run-taslp23-ploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with P-loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/03_train_effnet_ploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23sl run-taslp23-specloss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with Spectral loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/04_train_effnet_specloss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23pnpl run-taslp23-pnploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with PNP-loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/05_train_effnet_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

# Fine-tuning targets
rt23ft-pnpl run-taslp23-finetune-pnploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Fine-tuning EfficientNet with PNP-loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/06_train_effnet_finetune_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23ft-mss run-taslp23-finetune-mss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Fine-tuning EfficientNet with MSS loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/07_train_effnet_finetune_mss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23raw run-taslp23-pnploss-raw: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with PNP-loss raw parameters (TASLP23)..."
	source .venv/bin/activate && python taslp23/08_train_effnet_pnploss_raw.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23ft2-pnpl run-taslp23-finetune2-pnploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Alternative fine-tuning EfficientNet with PNP-loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/09_train_effnet_finetune_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23ft2-mss run-taslp23-finetune2-mss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Alternative fine-tuning EfficientNet with MSS loss (TASLP23)..."
	source .venv/bin/activate && python taslp23/10_train_effnet_finetune_mss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

# Complete TASLP23 training pipeline
rt23all run-taslp23-all: rt23pl rt23sl rt23pnpl rt23ft-pnpl rt23ft-mss

rt23o run-taslp23-observation:
	@echo "Starting TensorBoard for TASLP23 training..."
	@(cd /Users/jos/w/perceptual_neural_physical && \
	  . .venv/bin/activate && \
	  nohup tensorboard \
	    --logdir taslp23/outputs/taslp23/f_W/ploss_finetuneFalse_log-1_minmax-1_opt-adam_batch_size256_lr-0.001_init-test/logs \
	    > /dev/null 2>&1 &)
	@echo "TensorBoard running in background at http://localhost:6006"
	@echo "To stop TensorBoard later: pkill -f tensorboard"


# Conference order was
#   icassp23 (Oct 2022)
#   gretsi23 (2023)
#   taslp23 (June 2023)
#   mersenne24 (May 2024)
#   icassp25 (current)
# and we are using reverse order, so gretsi23 is next:

# -------------------- GRETSI-23  (AM Chirp Focused) -------------------

# File target - only runs if output doesn't exist
$(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	source .venv/bin/activate && python gretsi23/01_generate_audio.py $(SAVE_DIR)/gretsi23_audio

ag23 audio-gretsi23: $(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5

dg23 demo-gretsi23: $(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5
	@echo "Running GRETSI23 demo pipeline..."
	@mkdir -p $(SAVE_DIR)
	@echo "Note: This demo requires audio generation - see gretsi23/ directory for full pipeline"
	@echo "Example commands:"
	@echo "  cd gretsi23/ && python 01_generate_audio.py $(SAVE_DIR)/gretsi23_audio"
	@echo "  cd gretsi23/ && python 03_train_effnet_ploss.py $(SAVE_DIR)/gretsi23_models $(INIT_ID)"

# -------------------- ICASSP-23  (FTM Drums Focused) -------------------

# File target - only runs if output doesn't exist
icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	source .venv/bin/activate && python icassp23/01_generate_audio.py $(SAVE_DIR)/icassp23_audio

# Phony target aliases for convenience
ai23 audio-icassp23: icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5

# Run experiments (if data exists)
ri23pl run-icassp23-ploss:
	@echo "Training EfficientNet with P-loss (ICASSP23)..."
	source .venv/bin/activate && python icassp23/03_train_effnet_ploss.py $(SAVE_DIR)/icassp23 $(INIT_ID) $(BATCH_SIZE)

ri23pnpl run-icassp23-pnploss:
	@echo "Training EfficientNet with PNP-loss (ICASSP23)..."
	source .venv/bin/activate && python icassp23/06_train_effnet_pnploss.py $(SAVE_DIR)/icassp23 $(INIT_ID) $(BATCH_SIZE)

#================================== UTILITY MAKE TARGETS ==================================

# Data and output management
cro create-outputs:
	@mkdir -p $(SAVE_DIR)/{icassp23,gretsi23,taslp23,mersenne24,icassp25}
	@echo "Created output directories in $(SAVE_DIR)/"

# Cleanup
c clean:
	@echo "Cleaning build artifacts and cache files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete -print
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleanup completed"

clo clean-outputs:
	@echo "Removing output directories..."
	rm -rf $(SAVE_DIR)
	@echo "✓ Output directories removed"

# Environment info
info:
	@echo "Environment Information:"
	@echo "Python version: $$(source .venv/bin/activate && python --version)"
	@echo "Virtual environment: $$(if [ -d .venv ]; then echo 'Active'; else echo 'Not found'; fi)"
	@echo "Package installed: $$(source .venv/bin/activate && python -c 'import pnp_synth; print(pnp_synth.__file__)' 2>/dev/null || echo 'Not installed')"
	@echo "Save directory: $(SAVE_DIR)"
	@echo "Current experiment ID: $(INIT_ID)"

# Testing targets
test: test-import test-modules test-synthesis test-neural test-torch
	@echo "✓ All tests completed successfully"

test-import:
	@echo "Testing main package import..."
	source .venv/bin/activate && python -c "import pnp_synth; print('✓ pnp_synth imported successfully')"

test-modules:
	@echo "Testing module structure..."
	source .venv/bin/activate && python -c "from pnp_synth import utils; print('✓ utils module working')"
	source .venv/bin/activate && python -c "from pnp_synth.neural import cnn; print('✓ neural.cnn module working')"
	source .venv/bin/activate && python -c "from pnp_synth.perceptual import jtfs; print('✓ perceptual.jtfs module working')"

test-synthesis:
	@echo "Testing physical synthesis modules..."
	source .venv/bin/activate && python -c "from pnp_synth.physical import ftm; print('✓ FTM synthesis module working')"
	source .venv/bin/activate && python -c "from pnp_synth.physical import amchirp; print('✓ AM chirp synthesis module working')"

test-neural:
	@echo "Testing neural network modules..."
	source .venv/bin/activate && python -c "from pnp_synth.neural import cnn, forward, loss, optimizer; print('✓ Neural modules working')"

test-torch:
	@echo "Testing PyTorch setup..."
	source .venv/bin/activate && python -c "import torch; print(f'✓ PyTorch {torch.__version__} available')"
	source .venv/bin/activate && python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
	source .venv/bin/activate && python -c "import kymatio; print(f'✓ kymatio {kymatio.__version__} available')"

# Development tools
jupyter:
	@echo "Starting Jupyter Lab..."
	source .venv/bin/activate && jupyter lab

lint:
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8 linter..."; \
		source .venv/bin/activate && flake8 src/pnp_synth/ --max-line-length=100; \
	else \
		echo "flake8 not available - install with: uv pip install flake8"; \
	fi

format:
	@if command -v black >/dev/null 2>&1; then \
		echo "Formatting code with black..."; \
		source .venv/bin/activate && black src/pnp_synth/; \
	else \
		echo "black not available - install with: uv pip install black"; \
	fi
