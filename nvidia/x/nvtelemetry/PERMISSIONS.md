# Collection permissions and refusal boundaries

This inventory describes source at
`6ae90f61e619caf7b222de332c95be595dd8b928`. It separates access made by this
consumer from access made by an external exporter, scheduler adapter or injected
target. It is not a verified minimum capability set for an installed platform.
Unqualified filenames below are in this package. Paths beginning with `cmd/`
or `cudadiag/` are relative to `nvidia/x`; `ebpfhost/` paths are relative to
the repository root.

## Collection modes

| Mode and public entrypoint | Calls and resources | Denial or missing evidence | Qualification boundary |
| --- | --- | --- | --- |
| NVML-only `nvtelemetry serve` | `cmd/nvtelemetry/agent.go` selects explicit UUIDs through `nvml.Init`, `NewDeviceByUUID` and `Device.UUID`; `serve.go` uses the same selection for optional retained/durable serving. `nvml_sampler.go` calls vendor metadata and telemetry APIs. The runtime must supply a usable NVML library and selected device access. | An empty allowlist, failed initialization or UUID mismatch refuses acquisition. Optional telemetry query failures produce unavailable readings with a typed reason, not measured zero. | These branches do not install BPF probes or CUPTI injection. The vendor library's actual device opens/ioctls, UID/GID rules and native ABI need version-specific platform evidence; a template's device list is not that evidence. |
| Managed exporter through `FetchManaged` or managed `serve` | `managed_http.go` and `endpoint_http.go` issue a bounded GET using the supplied HTTP client/request; redirects are not followed. The CLI reads selected endpoint, header/bearer and CA files in `cmd/nvtelemetry/endpoint.go`. | Invalid source/configuration is refused; transport error, non-200 status, redirect and oversize response are errors. Missing data is not a healthy zero. | This client needs selected file reads and endpoint reachability. It does not acquire the exporter's GPU privileges. A DCGM Exporter per-process deployment is a separate producer/version/privilege cell; its hostPID, device and other requirements need the producer owner's evidence. |
| Host API/scheduler tracefs capture through `cudadiag.CollectHostEvidence` | `cudadiag/live_host.go` checks target process/thread membership. `ebpfhost/collector.go` creates a tracefs instance, writes tracing controls, reads `/proc/PID/task`, appends selected `uprobe_events` and enables events. | Missing deadline, invalid target, denied proc/tracefs access or setup failure returns an error. The receipt retains partial/unknown transport coverage. | This is separate from NVML-only collection. The host/kernel/namespace owner must admit the exact target and tracefs access; this inventory does not grant an attach or prescribe a universal capability combination. |
| Host BPF/perf uprobes through `ebpfhost/bpfprobe` | `generation_linux.go` reads `/sys/kernel/btf/vmlinux`. `attach_linux.go` uses BPF map/program APIs, reads uprobe PMU sysfs, calls `perf_event_open` and attach/enable ioctls. `ring_linux.go` inspects and mmaps a ring map. | Map/load/PMU/perf/ioctl/mmap failures are returned rather than replaced by fixture data. | BTF/sysfs/target-image visibility, BPF/perf authorization, namespace and LSM policy are separate from tracefs permissions. Source calls alone do not prove that CAP_BPF, CAP_PERFMON or any other fixed set is sufficient on a selected kernel. |
| CUPTI/injection/deep capture through `cudaprof capture` and the existing provider owners | `cmd/cudaprof/capture_launcher.go` constructs the target command and sets `CUDA_INJECTION64_PATH` or preload configuration for the selected agent. The target loads that library and produces capture artifacts. | A pre-existing CUDA injection setting is refused rather than silently replaced. Provider, process-scope, version, compatibility, start/drain and cleanup failures retain their capture-owner status. | Not a mode of the NVML DaemonSet below. Loading an agent in a launched target is not permission to attach to arbitrary processes. Exact target/library/device/artifact access and PC/PM/range/SASS mode admission remain with their existing owners. |

A governed sampler rejects a source device set that differs from the configured
allowlist before sampling. `CaptureRegistry` exclusivity is in-process; it is
not a cross-process lease or a container authorization boundary. An optional
unavailable counter can leave a successful source ready; a source error and
successful-publication age are separate from GPU or application health.

## Allocation and process identity

| Input or consumer | Actual access | Refusal and owner |
| --- | --- | --- |
| `ReadProcEntity` | Parses supplied `stat`, `status` and `cgroup` readers; performs no filesystem access itself. | Missing identity, malformed state or a dead/zombie snapshot refuses process identity. A cgroup path is not proof of a Pod UID, and proc start ticks are not an exec generation. |
| `ReadSlurmAttributionSnapshot` and the refresh CLI | The CLI uses `os.DirFS` for an explicitly configured root. The reader opens configured squeue, GPU-mapping and execution files, then only selected `ProcDir/PID/{stat,status,cgroup}`. It does not enumerate directories or read `environ`. | Denied/missing process files withhold that PID. A failed required shared file or another process read error fails the whole read. The source is validated before opening files. The external producer owns authentic scheduler acquisition and any required credentials. |
| `AttributionRefresher.Step/Resolve` | Reads through the supplied callback and updates the existing attribution table only after a successful read. | A failed cycle cannot create a fresh assignment. Previously valid entries remain subject to the existing MaxAge; expired source returns unknown. This is bounded snapshot freshness, not proof of current scheduler allocation. |
| `ReadPodResourcesJSON` | Parses a caller-supplied JSON reader and declared snapshot. It does not call a Kubernetes API or dial a kubelet socket. | The external adapter must provide authentic versioned PodResources data and its actual socket mount/UID rules, or exact API resources/verbs if it uses an API. Do not assign cluster-admin or get/list/watch permissions to this parser. |

Allocation authority, observed execution and physical device identity remain
separate. A supplied GPU index or `CUDA_VISIBLE_DEVICES` value is not a physical
ownership proof. MIG generation, MPS/time-slice sharing, process/boot epochs and
validity intervals must come from the selected authentic sources. Unknown
mappings do not authorize exact workload attribution.

## Serving, delivery, storage and secrets

| Configured capability | Calls and access | Refusal or limitation |
| --- | --- | --- |
| HTTP serving | `cmd/nvtelemetry/agent.go` uses `net.Listen` on the configured address. `config.go` requires allow_remote, TLS cert/key and bearer authentication for non-loopback serving. | Invalid config/key/token refuses setup; the bearer wrapper returns 401 for unauthorized requests. Network/firewall/namespace reachability and tenant authorization still need operator evidence. |
| Credentials and trust | The CLI reads bounded selected bearer/header/CA files; `tls.LoadX509KeyPair` reads the configured serving key/certificate. The Kubernetes template mounts configuration and credentials read-only. | Tokens are file references, not probe arguments. File-read and invalid-credential errors refuse the affected setup/request. Key readability, trust, expiry, mount policy and secret distribution need real platform checks. |
| Managed input / OTLP output | Managed input uses bounded HTTP requests; `delivery.go` POSTs to the selected endpoint with explicit timeout, retry and queue limits. | An HTTP success is not independent backend query acceptance or exactly-once delivery. Endpoint outage and queue loss do not become device-health facts. These outbound accesses are optional capabilities, not requirements of every local NVML scrape. |
| Retention, durable spool and attribution cache | Existing retention/delivery/cache APIs access explicitly configured owned directories; their file/lease/size/identity checks apply independently. | A source table cannot prove filesystem quota, crash durability or recovery. Writable storage is needed only when those capabilities are selected; preserve failed artifacts and actual owner/cleanup status. |
| Retained/exported identity labels | `CapturePolicy` and `SafeCaptureLabels` use explicit bounded selectors; arbitrary environment, command strings, credentials and model input labels are unsupported. | Selector tests do not prove complete cross-tenant isolation or detect every secret hidden inside a permitted identifier. Stacks/command collection needs its own explicit target and capture policy. |

## Platform templates

- The [DaemonSet template](../cmd/nvtelemetry/packaging/daemonset.yaml) is
  NVML-only. It drops capabilities, disables service-account token automounting,
  avoids host PID/network and hostPath device mounts, and uses runtime-injected
  NVML devices. Its runtime class, immutable image, TLS/Secret/ConfigMap and
  node-to-allocation-to-UUID mapping remain operator inputs. It makes no
  Kubernetes API calls and requests no RBAC role. BPF/CUPTI require a separate
  admitted mode; do not make this template privileged to enable them.
- The [systemd template](../cmd/nvtelemetry/packaging/nvtelemetry.service)
  currently grants CAP_PERFMON unconditionally and lists device-access
  directives. Its own comment says that capability is not proved or required
  for standard NVML operation. Those are actual template grants, not a verified
  minimum requirement or a least-privilege qualification. The packaging README
  explicitly leaves this service unqualified for NVML-only operation.
- Slurm and Kubernetes allocation acquisition remain external consumer work.
  Parser fixtures, service/YAML lint and a successful scrape do not prove an
  authentic job/Pod assignment, usable runtime device injection or access to
  the real PodResources socket. Managed exporter privilege variants must be
  recorded separately from this NVIDIA client and its NVML-only template.

## Evidence and remaining gates

`permission_refusal_test.go` exercises the real public Slurm reader/refresher
using injected filesystem errors. It distinguishes denied process evidence from
denied required source data and preserves unrelated process evidence and stale
source refusal. These are host behavior controls, not kernel, Slurm or Kubernetes
denied-access receipts. Existing governance, config, managed HTTP, probe and
packaging tests cover their narrower host contracts; the presence of a test is
not a current execution result. Historical receipts retain their original source,
producer, version and environment limits.

For each selected platform/mode, retain actual source/tool versions, configured
resources, allowed and denied operations, independent allocation/backend truth,
raw hashes, selected UUID/epochs, cleanup/loss, observer cost and owner/next
action. Device/proc/sys/BTF/cgroup/socket denials, RBAC, credential expiry/TLS,
real bare-metal/service-manager, Slurm and Kubernetes deployment, guest or
confidential restrictions and tenant isolation remain separate authentic gates.
No template, source inventory or host test closes them. Preserve HOLD until
those exact cells are admitted; neither remediation nor additional permissions
follow from a collection failure. Short platform checks do not close longer
operational soak.
