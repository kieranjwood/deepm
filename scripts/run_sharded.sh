#!/usr/bin/env bash
# Run one training config as N parallel shards (e.g. on a single rented GPU), then aggregate.
# Each shard runs the trials i with i % N == k (DEEPM_SHARD=k/N, see deepm/training/hp_tuner.py);
# a final run without DEEPM_SHARD finds every trial done and only writes all_runs/best_runs.
#
# Usage: scripts/run_sharded.sh <train_yaml> <arch> <n_shards> [threads_per_shard]
#   PY=.venv/bin/python scripts/run_sharded.sh deepm-gat DeePM 3 4
# Notes: a 4090 (24 GB) fits ~6-8 DeePM processes; set threads so shards x threads <= CPU quota
# (torch otherwise starts one thread per visible CPU, which on shared hosts is far too many).
# Multi-GPU boxes: shard k runs on GPU (GPU_BASE + k) % <number of GPUs> (GPU_BASE lets several configs
# started side by side spread over all GPUs).
set -euo pipefail
yaml=$1; arch=$2; n=$3; th=${4:-4}
PY=${PY:-.venv/bin/python}
export WANDB_MODE=offline OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)
[ "${ngpu:-0}" -ge 1 ] 2>/dev/null || ngpu=1
base=${GPU_BASE:-0}
mkdir -p logs
pids=()
for k in $(seq 0 $((n - 1))); do
  gpu=$(( (base + k) % ngpu ))
  CUDA_VISIBLE_DEVICES=$gpu DEEPM_SHARD="$k/$n" "$PY" -m deepm.training -r "$yaml" -a "$arch" \
    > "logs/train-$yaml.shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" != 0 ]; then echo "a shard of $yaml failed, see logs/train-$yaml.shard*.log" >&2; exit 1; fi
"$PY" -m deepm.training -r "$yaml" -a "$arch" > "logs/train-$yaml.aggregate.log" 2>&1
echo "$yaml: all shards done and aggregated"
