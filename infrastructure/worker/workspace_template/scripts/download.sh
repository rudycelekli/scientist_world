#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: TODO GB
# Estimated disk usage after extraction: TODO GB
#
# During reproduction, prefer downloading to shared directories
# (shared/datasets/, shared/models/) so other papers can reuse them.
# You do not need to run this script during reproduction — it exists
# for future portability. Write it to download to workspace-local
# paths (data/, checkpoints/) instead of the shared directories
# so a future user with just the workspace can run this script and get everything they need.

set -e

cd "$(dirname "$0")"

# TODO: implement downloads
# Examples:
#   wget -nc https://example.com/dataset.tar.gz -P data/
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('model/name', local_dir='checkpoints/')"
#   huggingface-cli download Qwen/Qwen3-1.7B --local-dir /checkpoints/Qwen3-1.7B
