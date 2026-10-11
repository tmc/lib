# BPF and GPU observations

These packages separate host observations, sampled device state, device activity,
and experimental GPU execution. They do not share a timing domain implicitly.

| Package | Purpose |
| --- | --- |
| `github.com/tmc/lib/ebpfhost` | Linux tracefs collection and ELF/process resolution |
| `github.com/tmc/lib/ebpfhost/bpfload` | Linux BPF program, map and BTF ownership |
| `github.com/tmc/lib/ebpfhost/bpfprobe` | Bounded host uprobes, returns and optional user stacks |
| `github.com/tmc/lib/ebpfhost/hostmetrics` | Bounded configured Linux node and Redfish source readers |
| `cudaprobe` | Decode supported CUDA Runtime and Driver API captures |
| `cudatracehost` | Own host probe collection and bounded entry/return correlation |
| `nvtelemetry` | NVML/DCGM sampling, versioned streams, identity, health, workload exports and accounting |
| `xidcatalog` | Pinned product-specific XID facts and advisory action names |
| `xidlog` | Bounded supplied kernel records with explicit identity and loss limits |
| `nvtelemetrybridge` | Publish coherent versioned snapshots for independent pinned-map consumers |
| `cudatrace` | Export host observations and telemetry in separate uncalibrated Perfetto artifacts |
| `cudaflightrecorder` | Bounded records and incident capture with explicit resource budgets |
| `cudadiag` | Diagnose only from admitted observations and controls |
| `cudabpf` | Validate experimental driver-policy capabilities and own supplied attachments |
| `cudabpfdev` | Verify a bounded device instruction dialect and emit/inject PTX |

Commands live in `x/cmd`. Collection is optional; generated CUDA and NVML
bindings do not depend on these packages.

The BPF CUDA host-probe backend implements Linux/amd64 System V and
Linux/arm64 AAPCS64 register and stack contracts. ARM64 has compiled UAPI,
argument-packing, decoder and assembler controls on DGX; privileged ARM64
probe attachment remains unqualified. ELF architecture and byte order are
validated before attachment. Other architectures are refused explicitly.

Host API intervals measure submission or waiting inside a library call. They
are not kernel execution intervals. Scheduler events describe host threads.
NVML query times are not sensor acquisition times. CUPTI activity requires an
admitted native record layout and separate capture ownership. Unknown clock
calibration, unsupported readings, incomplete coverage and transport losses
remain visible instead of becoming precise joins.

Host probes cover the threads present in the attachment snapshot. Threads
created later remain unobserved. Dynamic perf-uprobe inheritance is refused:
Linux can reinitialize inherited events in the target process and prevent
unprivileged thread creation. Snapshot coverage does not establish complete
process coverage. Frame-pointer stacks are bounded observations; omitted
frames and language stacks remain unknown.

The corrected host collector has a native receipt for six completed CUDA
Driver APIs on a Linux/amd64 L4 workload: allocation, free, both copy
directions, launch and synchronization. The workload passed its numerical
oracle, all twelve probe records were delivered, and cleanup released the GPU
lease. This receipt is for source `8a6fbf68b` and its recorded executable and
fixture hashes. It does not qualify H100, future-thread attachment, complete
lifecycle capture, scheduler clock joins or observer overhead. Earlier failed
inheritance receipts remain failures; the correction was qualified separately.

`cudabpf` describes pinned experimental `gpu_ext` interfaces. Stock kernels and
NVIDIA drivers do not expose these hooks. Its Go policies are host models;
they are not kernel programs. Discovery and attachment refuse missing or
contradictory capability evidence.

`cudabpfdev` accepts its documented bounded instruction dialect, not arbitrary
Linux eBPF or general bpftime input. The native fixtures have numerical,
counter, guard and private-module isolation receipts on GB10 with CUDA 13.
See [device qualification](cudabpfdev/QUALIFICATION.md) for the candidate pins
and limits. Those receipts do not qualify H100, active driver policies,
shared staging, concurrent hardware execution or collection overhead.

Synthetic diagnosis/evaluation examples exercise contracts. They cannot admit
physical timing, comparative performance or production-readiness claims. The
native demonstration executes a numerical CUDA workload and reports missing
activity, scheduler, mapping and overhead gates separately.

Telemetry adapters consume pinned native interfaces or bounded documented
exports. Kubernetes and Slurm assignments require supplied authoritative
identity and process-generation evidence. Node observations require an explicit
node/device join. Workload readers cover Triton, vLLM, Ray, PyTorch, W&B, MLflow
and ClearML; source-derived fixtures do not establish authenticated deployment
compatibility with every provider. See [workload contracts](nvtelemetry/WORKLOADS.md),
[managed sources](nvtelemetry/MANAGED.md) and
[governed collection](nvtelemetry/operations.md).

The native DGX NVML receipt compares a frozen sampler and health reader with
independent compiled C reads. Unsupported memory, ECC and remap fields remain
unavailable. The two-L4 NCCL receipt independently checks normal and delayed
AllReduce values and records actual RAS identities. Neither receipt establishes
calibrated device timelines, sensor freshness, H100 compatibility or observer
overhead. Exact source, executable and provider hashes remain in the receipts;
later source or history changes do not transfer their qualification.

Prometheus and OTLP exports preserve bounded evidence metadata. OTLP retains
exact integer strings; Prometheus floating-point storage can round values above
2^53. Workload delta sums and histograms require a receiver that supports their
temporality. Diagnostic logs are a separate signal. An accepted local receiver
pipeline does not establish SaaS backend parity or production delivery.
