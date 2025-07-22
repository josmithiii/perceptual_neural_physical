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
	cd icassp25/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/icassp25

# ICASSP25 data generation using FTM modal drum synthesis
ai25 audio-icassp25: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5
	@echo "✓ ICASSP25 audio data generated"

# Test audio generation (first 100 samples per fold)
ai25test audio-icassp25-test:
	@echo "Generating ICASSP25 test audio data (100 samples per fold)..."
	@mkdir -p $(SAVE_DIR)/icassp25
	cd icassp25/ && source ../.venv/bin/activate && python 01_generate_audio_test.py $(SAVE_DIR)/icassp25
	@echo "✓ ICASSP25 test audio data generated"

# Compute PNP Jacobian matrices
Mi25 M-icassp25: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5
	@echo "Computing PNP Jacobian matrices for ICASSP25..."
	cd icassp25/ && source ../.venv/bin/activate && python 01b_compute_pnp_jacobian.py $(SAVE_DIR)/icassp25

# Training targets
ri25pl run-icassp25-ploss: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5
	@echo "Training EfficientNet with P-loss (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/icassp25 $(INIT_ID) 1 1 adam b0 $(BATCH_SIZE)

ri25pnpl run-icassp25-pnploss: $(SAVE_DIR)/icassp25/x/ftm_train_audio.h5 Mi25
	@echo "Training EfficientNet with PNP-loss (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 05_train_effnet_pnploss.py $(SAVE_DIR)/icassp25 $(INIT_ID) 1 1 adam b0 $(BATCH_SIZE)

ri25all run-icassp25-all: ri25pl ri25pnpl

# Evaluation targets
ei25pl eval-icassp25-ploss:
	@echo "Evaluating P-loss model (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 01_eval_ploss.py b0 adam $(INIT_ID) > ei25pl_log_$$(date +%Y-%m-%d).txt 2>&1 &
	@echo "Background evaluation started. Check log: icassp25/ei25pl_log_$$(date +%Y-%m-%d).txt"

ei25pnp eval-icassp25-pnp:
	@echo "Evaluating PNP model (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 02_eval_pnp.py b0 adam 0

# Gradient analysis targets
ei25grad eval-icassp25-grad:
	@echo "Evaluating gradients (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 06_eval_grad.py adam ploss b0 32

ei25grad_ft eval-icassp25-grad-finetune:
	@echo "Evaluating gradients for fine-tuning (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 08_eval_grad_finetune.py adam

ei25grad_original eval-icassp25-grad-original:
	@echo "Evaluating gradients (ICASSP25) - original config..."
	cd icassp25/ && source ../.venv/bin/activate && python 06_eval_grad.py adam ploss b0 256

# Audio comparison and analysis
ei25audio eval-icassp25-audio-comparison:
	@echo "Generating audio comparison (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python audio_comparison.py $(SAVE_DIR)/icassp25/f_W/b0_ploss_finetuneFalse_log-1_minmax-1_opt-adam_batch_size$(BATCH_SIZE)_lr-0.001_init-$(INIT_ID) --num_samples=10 --output_dir=./audio_comparison

ei25html eval-icassp25-html:
	@echo "Generating HTML comparison interface (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python generate_comparison_html.py ./audio_comparison/analysis.json --output=./audio_comparison/comparison.html
	@echo "✓ Interactive HTML comparison generated at icassp25/audio_comparison/comparison.html"

# -------------------- TASLP-23  -------------------

# File target - only runs if output doesn't exist  
$(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	cd taslp23/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/taslp23

at23 audio-taslp23: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "✓ TASLP23 audio data generated"

cat23 clean-audio-taslp23:
	-/bin/rm -rf $(SAVE_DIR)/taslp23/x/

# Compute PNP Jacobians for TASLP23
jt23 jacobian-taslp23: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Computing PNP Jacobian matrices for TASLP23..."
	cd taslp23/ && source ../.venv/bin/activate && python 02_compute_pnp_jacobian.py $(SAVE_DIR)/taslp23 0 1000 1 0 1

# Compute LMA step matrices
Mt23 M-taslp23: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Computing LMA step matrices for TASLP23..."
	cd taslp23/ && source ../.venv/bin/activate && python 12_compute_lmastep.py $(SAVE_DIR)/taslp23 0 1000

# Merge H5 files
mht23 merge-h5-taslp23:
	@echo "Merging H5 files for TASLP23..."
	cd taslp23/ && source ../.venv/bin/activate && python 13_merge_h5file.py

# Basic training targets
rt23pl run-taslp23-ploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with P-loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23sl run-taslp23-specloss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with Spectral loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 04_train_effnet_specloss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23pnpl run-taslp23-pnploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with PNP-loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 05_train_effnet_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

# Fine-tuning targets
rt23ft-pnpl run-taslp23-finetune-pnploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Fine-tuning EfficientNet with PNP-loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 06_train_effnet_finetune_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23ft-mss run-taslp23-finetune-mss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Fine-tuning EfficientNet with MSS loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 07_train_effnet_finetune_mss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23raw run-taslp23-pnploss-raw: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Training EfficientNet with PNP-loss raw parameters (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 08_train_effnet_pnploss_raw.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23ft2-pnpl run-taslp23-finetune2-pnploss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Alternative fine-tuning EfficientNet with PNP-loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 09_train_effnet_finetune_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23ft2-mss run-taslp23-finetune2-mss: $(SAVE_DIR)/taslp23/x/taslp23_train_audio.h5
	@echo "Alternative fine-tuning EfficientNet with MSS loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 10_train_effnet_finetune_mss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

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

# -------------------- ICASSP-23  -------------------

# File target - only runs if output doesn't exist
icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	cd ./icassp23/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/icassp23_audio

# Phony target aliases for convenience
ai23 audio-icassp23: icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5

# Run experiments (if data exists)
ri23pl run-icassp23-ploss:
	@echo "Training EfficientNet with P-loss (ICASSP23)..."
	cd icassp23/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/icassp23 $(INIT_ID) $(BATCH_SIZE)

ri23pnpl run-icassp23-pnploss:
	@echo "Training EfficientNet with PNP-loss (ICASSP23)..."
	cd icassp23/ && source ../.venv/bin/activate && python 06_train_effnet_pnploss.py $(SAVE_DIR)/icassp23 $(INIT_ID) $(BATCH_SIZE)


# -------------------- GRETSI-23  -------------------

# File target - only runs if output doesn't exist
$(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	cd ./gretsi23/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/gretsi23_audio

ag23 audio-gretsi23: $(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5

dg23 demo-gretsi23: $(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5
	@echo "Running GRETSI23 demo pipeline..."
	@mkdir -p $(SAVE_DIR)
	@echo "Note: This demo requires audio generation - see gretsi23/ directory for full pipeline"
	@echo "Example commands:"
	@echo "  cd gretsi23/ && python 01_generate_audio.py $(SAVE_DIR)/gretsi23_audio"
	@echo "  cd gretsi23/ && python 03_train_effnet_ploss.py $(SAVE_DIR)/gretsi23_models $(INIT_ID)"

# -------------------- UTILITY MAKE TARGETS  -------------------

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
