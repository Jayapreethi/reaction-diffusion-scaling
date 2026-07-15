#!/bin/bash
# SLURM GPU Comparison Job
#SBATCH --job-name=gpu_comparison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gpu_comparison.log

# Configuration (set these as needed)
GRID_SIZE=${GRID_SIZE:-128}
TIMESTEPS=${TIMESTEPS:-50}
NUM_PARTITIONS=${NUM_PARTITIONS:-1}
OUTPUT_FILE="outputs/gpu_results_${GRID_SIZE}x${GRID_SIZE}.json"

# Load CUDA modules if available (cluster-specific)
module load cuda 2>/dev/null || true
module load nvidia 2>/dev/null || true

# Assume script is in project root or adjust path as needed
cd "$(dirname "$0")"

# Run benchmark
python3 scripts/gpu_comparison.py \
    --grid-size "$GRID_SIZE" \
    --timesteps "$TIMESTEPS" \
    --num-partitions "$NUM_PARTITIONS" \
    --output-json "$OUTPUT_FILE"

echo "Job completed. Results saved to: $OUTPUT_FILE"
