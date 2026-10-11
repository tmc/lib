# NVML-only capture profiles

These scripts and Pod manifest use the actual `record -device-uuids` allowlist.
Empty selection means all GPUs in the CLI, so these profiles require explicit
UUIDs. NVML UUID handles are resolved and verified before constructing the
sampler; unrelated devices are not enumerated. CUDA_VISIBLE_DEVICES alone does
not constrain NVML. Verify allocation membership outside the collector.

Build `go build -o nvtelemetry ./x/cmd/nvtelemetry` from the nvidia module and
place that binary on PATH. Set NVTELEMETRY_DEVICE_UUIDS and NVTELEMETRY_OUTPUT,
then run `sh bare-metal.sh`, or submit `slurm.sh` with the stated GPU allocation.
For Kubernetes supply an immutable built image and the scheduler-assigned UUID
before applying the example in an explicitly selected namespace. This package
does not apply manifests, allocate GPUs, discover Slurm membership or install
the NVIDIA device plugin. Placeholder values are intentionally unusable.

| Backend | Devices, mounts and access | Privileges |
| --- | --- | --- |
| These NVML-only profiles | User-readable NVML library and allocated NVIDIA device nodes supplied by host/runtime; private writable capture directory | No CAP_BPF, CAP_PERFMON, CAP_SYS_ADMIN, CUPTI injection, host PID/network namespace, host proc/sys/BTF/cgroup mounts or Kubernetes API/RBAC/token required |
| BPF or CUPTI modes | Not enabled by these profiles | Admit the selected backend separately; do not add broad privileges to this NVML Pod |

The 16 MiB stream budget, record/buffer limits, fixed duration and Pod volume
limit bound this capture. Budget exhaustion is a visible failure/incomplete
stream; it is not a complete diagnostic result. No automatic upload, labels,
environment values or stack/command capture is enabled. Host/native permissions
and real Slurm/Kubernetes/device-plugin versions remain qualification gates.
The manifest has no API readiness probe: consumers must check the completion
manifest, freshness and explicit source state rather than Pod liveness alone.

Library consumers can use CaptureRegistry and Retention for explicit policy,
bounded owned-file deletion and conservative restart reconciliation. These
profiles do not pretend the CLI invokes those additional library APIs.
