#!/bin/bash

# setup.sh - Set up uv environment for perceptual_neural_physical project
# Based on installation requirements from README.md

set -e  # Exit on any error

echo "Setting up uv environment for perceptual_neural_physical..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create and activate virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv

# Activate the virtual environment
source .venv/bin/activate
echo "Virtual environment activated"

# Install the project in editable mode with dependencies
echo "Installing project dependencies..."
uv pip install -e .

# Install training dependencies
echo "Installing training dependencies..."
uv pip install tensorboard

# Install kymatio (try GPU version, fallback to standard)
echo "Installing kymatio..."
if ! uv pip install kymatio; then
    echo "Standard kymatio installation failed, trying GPU version..."
    if [ ! -d "jtfs-gpu" ]; then
        git clone https://github.com/cyrusasfa/jtfs-gpu.git
    fi
    cd jtfs-gpu
    uv pip install -e .
    cd ..
else
    echo "Standard kymatio installed successfully"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation, you can test importing the package:"
echo "  python -c 'import pnp_synth; print(\"pnp_synth imported successfully\")'"
