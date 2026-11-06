#!/bin/bash
# Example run.sh — update this file before each job submission.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

# Example: train a model
python method/train.py --epochs 10 --lr 0.001

# Example: evaluate
./evaluate.sh checkpoints/best.pt
