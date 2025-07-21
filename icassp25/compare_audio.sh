#!/bin/bash

# Complete Audio Comparison Workflow
# Usage: ./compare_audio.sh <model_dir> [num_samples] [output_dir]

set -e  # Exit on error

MODEL_DIR="$1"
NUM_SAMPLES="${2:-10}"
OUTPUT_DIR="${3:-./audio_comparison}"

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <model_dir> [num_samples] [output_dir]"
    echo "Example: $0 outputs/icassp25/f_W/b0_ploss_..._init-test 5"
    exit 1
fi

echo "🎵 PNP Audio Comparison Workflow"
echo "================================"
echo "Model directory: $MODEL_DIR"
echo "Number of samples: $NUM_SAMPLES" 
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Activate virtual environment
source ../.venv/bin/activate

echo "Step 1: Generating audio comparisons..."
python audio_comparison.py "$MODEL_DIR" --num_samples="$NUM_SAMPLES" --output_dir="$OUTPUT_DIR"

echo ""
echo "Step 2: Creating interactive HTML interface..."
python generate_comparison_html.py "$OUTPUT_DIR/analysis.json" --output="$OUTPUT_DIR/comparison.html"

echo ""
echo "✅ Audio comparison complete!"
echo ""
echo "📁 Generated files:"
echo "   - Audio files: $OUTPUT_DIR/*.wav"
echo "   - Analysis: $OUTPUT_DIR/analysis.json" 
echo "   - Interactive interface: $OUTPUT_DIR/comparison.html"
echo ""
echo "🌐 Open the comparison interface:"
echo "   open $OUTPUT_DIR/comparison.html"
echo ""
echo "🎧 To listen to target vs reconstruction:"
echo "   1. Open comparison.html in your web browser"
echo "   2. Use the audio players to compare target vs reconstruction"
echo "   3. Navigate between samples using arrow keys or buttons"
echo "   4. Check parameter values and prediction errors"