#!/bin/sh
# Run as the allocated GPU's unprivileged user; no sudo or BPF attachment.
set -eu
: "${NVTELEMETRY_DEVICE_UUIDS:?set the exact allocated GPU or MIG UUID allowlist}"
: "${NVTELEMETRY_OUTPUT:?set a task-owned output path}"
umask 077
exec nvtelemetry record -device-uuids "$NVTELEMETRY_DEVICE_UUIDS" \
  -output "$NVTELEMETRY_OUTPUT" -duration 60s -interval 100ms \
  -budget 10000 -byte-budget 16777216 -buffer-records 1000
