#!/bin/sh
# Submit with sbatch --gres=gpu:1 --time=00:02:00 slurm.sh after setting variables.
# Do not derive an NVML allowlist from CUDA_VISIBLE_DEVICES indexes.
set -eu
: "${SLURM_JOB_ID:?run inside a Slurm GPU allocation}"
: "${NVTELEMETRY_DEVICE_UUIDS:?set UUIDs verified against this job allocation}"
: "${NVTELEMETRY_OUTPUT:?set a private job-owned output path}"
umask 077
exec nvtelemetry record -device-uuids "$NVTELEMETRY_DEVICE_UUIDS" \
  -output "$NVTELEMETRY_OUTPUT" -duration 60s -interval 100ms \
  -budget 10000 -byte-budget 16777216 -buffer-records 1000
