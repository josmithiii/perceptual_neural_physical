# Makefile for Perceptual Neural Physical Sound Matching
# Research codebase with experiment pipelines and development workflows

.PHONY: help setup install test clean lint format jupyter experiments

# Default target
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
	@echo "Experiments (Examples):"
	@echo "  di23/demo-icassp23  - Run basic ICASSP23 pipeline demo"
	@echo "  demo-gretsi23       - Run basic GRETSI23 pipeline demo"
	@echo "  demo-taslp23        - Run basic TASLP23 pipeline demo"
	@echo ""
	@echo "Training:"
	@echo "  ri23pl         - Train EfficientNet with P-loss (ICASSP23)"
	@echo "  ri23pnpl       - Train EfficientNet with PNP-loss (ICASSP23)"
	@echo "  rt23pl         - Train EfficientNet with P-loss (TASLP23)"
	@echo "  rt23pnpl       - Train EfficientNet with PNP-loss (TASLP23)"
	@echo "  ri25pl         - Train EfficientNet with P-loss (ICASSP25)"
	@echo "  ri25pnpl       - Train EfficientNet with PNP-loss (ICASSP25)"
	@echo ""
	@echo "Evaluation:"
	@echo "  ei25pl         - Evaluate P-loss model"
	@echo "  ei25pnp        - Evaluate PNP model"
	@echo "  ei25grad       - Evaluate gradients (batch_size=32)"
	@echo "  ei25grad_original - Evaluate gradients (original config, batch_size=256)"
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

# Experiment demos (basic pipeline examples)

# File target - only runs if output doesn't exist
icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	(cd ./icassp23/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/icassp23_audio)

# Phony target aliases for convenience
ai23 audio-icassp23: icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5

di23 demo-icassp23: icassp23/outputs/icassp23_audio/x/icassp23_train_audio.h5
	@echo "Running ICASSP23 demo pipeline..."
	(cd ./icassp23/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/icassp23_models $(INIT_ID) $(BATCH_SIZE))

# File target - only runs if output doesn't exist
$(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	(cd ./gretsi23/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/gretsi23_audio)

ag23 audio-gretsi23: $(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5

dg23 demo-gretsi23: $(SAVE_DIR)/gretsi23_audio/gretsi23_train_audio.h5
	@echo "Running GRETSI23 demo pipeline..."
	@mkdir -p $(SAVE_DIR)
	@echo "Note: This demo requires audio generation - see gretsi23/ directory for full pipeline"
	@echo "Example commands:"
	@echo "  cd gretsi23/ && python 01_generate_audio.py $(SAVE_DIR)/gretsi23_audio"
	@echo "  cd gretsi23/ && python 03_train_effnet_ploss.py $(SAVE_DIR)/gretsi23_models $(INIT_ID)"

# File target - only runs if output doesn't exist  
$(SAVE_DIR)/taslp23_audio/x/taslp23_train_audio.h5:
	@mkdir -p $(SAVE_DIR)
	(cd taslp23/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/taslp23_audio)

at23 audio-taslp23: $(SAVE_DIR)/taslp23_audio/x/taslp23_train_audio.h5
	@echo "✓ TASLP23 audio data generated"

dt23 demo-taslp23: $(SAVE_DIR)/taslp23_audio/x/taslp23_train_audio.h5
	@echo "Running TASLP23 demo pipeline..."
	@mkdir -p $(SAVE_DIR)
	@echo "Note: This demo requires audio generation - see taslp23/ directory for full pipeline"
	@echo "Example commands:"
	@echo "  cd taslp23/ && python 01_generate_audio.py $(SAVE_DIR)/taslp23_audio"
	(cd taslp23/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/taslp23_models $(INIT_ID) 1 1 adam ftm)


# Experiment runners (if data exists)
ri23pl run-icassp23-ploss:
	@echo "Training EfficientNet with P-loss (ICASSP23)..."
	cd icassp23/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/icassp23 $(INIT_ID) $(BATCH_SIZE)

ri23pnpl run-icassp23-pnploss:
	@echo "Training EfficientNet with PNP-loss (ICASSP23)..."
	cd icassp23/ && source ../.venv/bin/activate && python 06_train_effnet_pnploss.py $(SAVE_DIR)/icassp23 $(INIT_ID) $(BATCH_SIZE)

rt23pl run-taslp23-ploss:
	@echo "Training EfficientNet with P-loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm

rt23pnpl run-taslp23-pnploss:
	@echo "Training EfficientNet with PNP-loss (TASLP23)..."
	cd taslp23/ && source ../.venv/bin/activate && python 05_train_effnet_pnploss.py $(SAVE_DIR)/taslp23 $(INIT_ID) 1 1 adam ftm


# ICASSP25 data generation using FTM modal drum synthesis
ai25 audio-icassp25:
	@echo "Generating ICASSP25 audio data using FTM synthesis..."
	@mkdir -p $(SAVE_DIR)/icassp25
	cd icassp25/ && source ../.venv/bin/activate && python 01_generate_audio.py $(SAVE_DIR)/icassp25
	@echo "✓ ICASSP25 audio data generated"

Mi25 M-icassp25:
	@echo "Computing PNP Jacobian matrices for ICASSP25..."
	cd icassp25/ && source ../.venv/bin/activate && python 01b_compute_pnp_jacobian.py $(SAVE_DIR)/icassp25

# ICASSP25 experiment runners
ri25pl run-icassp25-ploss:
	@echo "Training EfficientNet with P-loss (ICASSP25)..."
	@mkdir -p $(SAVE_DIR)/icassp25
	cd icassp25/ && source ../.venv/bin/activate && python 03_train_effnet_ploss.py $(SAVE_DIR)/icassp25 $(INIT_ID) 1 1 adam b0 $(BATCH_SIZE)

ri25pnpl run-icassp25-pnploss:
	@echo "Training EfficientNet with PNP-loss (ICASSP25)..."
	@mkdir -p $(SAVE_DIR)/icassp25
	cd icassp25/ && source ../.venv/bin/activate && python 05_train_effnet_pnploss.py $(SAVE_DIR)/icassp25 $(INIT_ID) 1 1 adam b0 $(BATCH_SIZE)

#ri25all: ri23pl ri23pnpl ri25pl ri25pnpl

ri25all: ri23pnpl ri25pl ri25pnpl

ei25pl eval-icassp25-ploss:
	@echo "Evaluating P-loss model (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 01_eval_ploss.py b0 adam test > ei25pl_log_`shortdate`.txt 2>&1 &
	tail -f icassp25/ei25pl_log_`shortdate`.txt

ei25pnp eval-icassp25-pnp:
	@echo "Evaluating PNP model (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 02_eval_pnp.py b0 adam 0

ei25grad eval-icassp25-grad:
	@echo "Evaluating gradients (ICASSP25)..."
	cd icassp25/ && source ../.venv/bin/activate && python 06_eval_grad.py adam ploss b0 32

ei25grad_original eval-icassp25-grad-original:
	@echo "Evaluating gradients (ICASSP25) - original config..."
	cd icassp25/ && source ../.venv/bin/activate && python 06_eval_grad.py adam ploss b0 256

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
