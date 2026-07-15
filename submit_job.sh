#!/bin/bash
# Fisher-KPP GPU Comparison SLURM Job
#SBATCH --job-name=gpu_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gpu_comparison_%j.log
#SBATCH --error=gpu_comparison_%j.err

# Configuration (set as environment variables before sbatch)
GRID_SIZE=${GRID_SIZE:-128}
TIMESTEPS=${TIMESTEPS:-50}
NUM_PARTITIONS=${NUM_PARTITIONS:-1}

# Optional: load cuda module if available (cluster-specific)
module load cuda 2>/dev/null || true

# Determine project root (assumes script is in project directory)
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Fisher-KPP GPU Comparison Experiment"
echo "=========================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Grid size: ${GRID_SIZE}x${GRID_SIZE}"
echo "Timesteps: $TIMESTEPS"
echo ""

# Show GPU info
nvidia-smi
echo ""

python3 scripts/gpu_comparison.py --grid-size 128 --timesteps 50 --num-partitions 1 --output-json outputs/gpu_results.json

echo ""
echo "Results saved to outputs/gpu_results.json"
