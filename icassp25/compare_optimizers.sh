#!/bin/bash
# Compare Adam vs Sophia optimizers for percussion parameter estimation
# Usage: ./compare_optimizers.sh <save_dir> <init_id>

set -e  # Exit on error

SAVE_DIR=${1:-"outputs/icassp25"}
INIT_ID=${2:-"test"}
DATE=$(date +%Y-%m-%d)
INIT_ID="${INIT_ID}_${DATE}"  # Append date for easy filtering
MINMAX=1          # Use MinMax scaling [-1, 1]
LOGSCALE=1        # Use log-scale for frequency parameters
EFF_TYPE="b0"     # EfficientNet-B0 architecture
BATCH_SIZE=32     # Batch size

echo "============================================"
echo "Optimizer Comparison Experiment"
echo "============================================"
echo "Date: $DATE"
echo "Save directory: $SAVE_DIR"
echo "Init ID: $INIT_ID"
echo "EfficientNet: $EFF_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "MinMax scaling: $MINMAX"
echo "Log-scale: $LOGSCALE"
echo ""

# Train with Adam optimizer
echo "============================================"
echo "Training with ADAM optimizer..."
echo "============================================"
python icassp25/03_train_effnet_ploss.py \
    "$SAVE_DIR" \
    "${INIT_ID}_adam" \
    "$MINMAX" \
    "$LOGSCALE" \
    "adam" \
    "$EFF_TYPE" \
    "$BATCH_SIZE"

# Train with Sophia optimizer
echo ""
echo "============================================"
echo "Training with SOPHIA optimizer..."
echo "============================================"
python icassp25/03_train_effnet_ploss.py \
    "$SAVE_DIR" \
    "${INIT_ID}_sophia" \
    "$MINMAX" \
    "$LOGSCALE" \
    "sophia" \
    "$EFF_TYPE" \
    "$BATCH_SIZE"

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  Adam:   $SAVE_DIR/f_W/${EFF_TYPE}_ploss_finetuneFalse_log-${LOGSCALE}_minmax-${MINMAX}_opt-adam_batch_size${BATCH_SIZE}_lr-0.001_init-${INIT_ID}_adam"
echo "  Sophia: $SAVE_DIR/f_W/${EFF_TYPE}_ploss_finetuneFalse_log-${LOGSCALE}_minmax-${MINMAX}_opt-sophia_batch_size${BATCH_SIZE}_lr-0.001_init-${INIT_ID}_sophia"
echo ""
echo "To compare TensorBoard logs:"
echo "  tensorboard --logdir=$SAVE_DIR/f_W/"
echo ""
echo "To filter by date in TensorBoard:"
echo "  Use regex filter: $DATE"
echo ""
echo "To analyze best checkpoints:"
echo "  python print_best_checkpoint.py <model_dir>"
