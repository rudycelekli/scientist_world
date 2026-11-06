#!/bin/bash
# Submit all test workloads to a running Hive cluster.
# Usage: bash tests/live/submit_workloads.sh
#
# Prerequisites: a running Hive cluster (`hive start ...`); GPU workloads need torch on the workers
# View results:  hive jobs / hive logs <job_id> / hive serve

set -e

# Resolve absolute path to repo root (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
W="$REPO_DIR/tests/workloads"

TOTAL=18
echo "=== Submitting $TOTAL test workloads to Hive ==="
echo "  Workload dir: $W"
echo ""

# 1. stdout capture test (verifies stdout.log works)
echo "[1/$TOTAL] stdout capture test..."
hive submit "echo 'STDOUT_MARKER: hello from hive'"
echo ""

# 2. stderr capture test (verifies stderr.log works)
echo "[2/$TOTAL] stderr capture test..."
hive submit "echo 'STDERR_MARKER: error output test' >&2 && exit 1"
echo ""

# 3. Quick sleep job (tests basic async queue + output capture)
echo "[3/$TOTAL] Sleep job (5s)..."
hive submit "python $W/sleep_job.py --seconds 5"
echo ""

# 4. Sleep job that intentionally fails (tests error capture)
echo "[4/$TOTAL] Failing sleep job..."
hive submit "python $W/sleep_job.py --seconds 2 --fail"
echo ""

# 5. GPU matmul benchmark (tests single-GPU compute)
echo "[5/$TOTAL] GPU matmul benchmark..."
hive submit "python $W/matmul_bench.py --size 4096 --iters 200"
echo ""

# 6. CNN training on random data (tests full training pipeline)
echo "[6/$TOTAL] CNN training (5 epochs)..."
hive submit "python $W/mnist_train.py --epochs 5"
echo ""

# 7. Multi-GPU DDP training (tests multi-GPU job submission)
echo "[7/$TOTAL] Multi-GPU DDP training (2 GPUs)..."
hive submit "torchrun --nproc_per_node=2 $W/multi_gpu_ddp.py" --gpus 2
echo ""

# 8. Transformer LM training (tests attention layers + large model)
echo "[8/$TOTAL] Transformer LM training (10 epochs)..."
hive submit "python $W/transformer_train.py --epochs 10"
echo ""

# 9. Mixed-precision AMP training (tests FP16/BF16 + GradScaler)
echo "[9/$TOTAL] AMP mixed-precision training..."
hive submit "python $W/amp_training.py --epochs 10"
echo ""

# 10. Checkpoint save/load (tests filesystem I/O + model state)
echo "[10/$TOTAL] Checkpoint I/O test..."
hive submit "python $W/checkpoint_io.py --epochs 5"
echo ""

# 11. GPU memory bandwidth (tests host-device transfers)
echo "[11/$TOTAL] Memory bandwidth test..."
hive submit "python $W/mem_bandwidth.py --size-gb 1 --iters 10"
echo ""

# 12. ResNet-like CNN (tests conv layers, BatchNorm, residual connections)
echo "[12/$TOTAL] ResNet CNN training (10 epochs)..."
hive submit "python $W/conv_resnet.py --epochs 10"
echo ""

# 13. Gradient accumulation (tests simulated large-batch training)
echo "[13/$TOTAL] Gradient accumulation training..."
hive submit "python $W/grad_accumulation.py --epochs 10 --accum-steps 8"
echo ""

# 14. Inference benchmark (tests forward-pass throughput)
echo "[14/$TOTAL] Inference benchmark (transformer)..."
hive submit "python $W/inference_bench.py --model transformer"
echo ""

# 15. VAE training (tests encoder-decoder + KL loss)
echo "[15/$TOTAL] VAE training (15 epochs)..."
hive submit "python $W/vae_train.py --epochs 15"
echo ""

# --- CPU-only jobs (no GPU required, --gpus 0) ---

# 16. CPU-only sleep job (tests CPU-only dispatch)
echo "[16/$TOTAL] CPU-only sleep job (5s)..."
hive submit "python $W/sleep_job.py --seconds 5" --gpus 0 --cpus 4
echo ""

# 17. CPU stress test (tests multi-threaded CPU workload)
echo "[17/$TOTAL] CPU stress test (10s, 4 threads)..."
hive submit "python $W/cpu_stress.py --seconds 10 --threads 4" --gpus 0 --cpus 4
echo ""

# 18. CPU-only stdout test (verifies CPU job output capture)
echo "[18/$TOTAL] CPU-only stdout test..."
hive submit "echo 'CPU_ONLY_MARKER: hello from cpu node'" --gpus 0 --cpus 1
echo ""

echo "=== All $TOTAL jobs submitted ==="
echo "Monitor with:  hive jobs"
echo "View logs:     hive logs <job_id>"
echo "Dashboard:     hive serve"
