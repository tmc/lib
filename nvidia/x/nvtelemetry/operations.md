# Read-only health and governed collection

`HealthSnapshot` version 1 is a separate bounded incident bundle. It does not
change telemetry stream version 2. `ReadHealthSnapshot` and `WriteHealthSnapshot`
validate source/device identities, canonical field units, topology validity,
query bounds, exact integer copies, duplicated evidence, lost records and byte
and record budgets. JSON struct field names must match their tags exactly;
duplicate keys, aliases, null scalars, unknown fields and trailing documents
are rejected. Dynamic map keys remain case-sensitive.

Reader availability and device state are separate. A failed RDMA source does
not imply GPU loss. A successful zero utilization reading is not an outage.
`QueryStale` measures age of the last successful query in the bundle's clock
domain/epoch; it never establishes sensor freshness. `Complete` describes
recorded bundle completeness, including explicitly unavailable fields, not a
health judgment or proof that every hardware field is implemented.

`TopologyEdge` has directed endpoints, scope, source, topology epoch and a
validity interval. An available edge must join two present devices in this
bundle. A reported nominal capability is not a measured link rate. Reconfigure
by producing a new bundle and topology epoch; this API does not cache edges or
carry missing values forward. Raw fields, vendor judgments and diagnostic
inferences have separate evidence classes. No API executes remediation.

## Source support

The following table distinguishes implemented readers from fields accepted
by the composition contract. A composable field is not a native reader receipt.

| Source/field | Library path | Portable receipt | Remaining proof or implementation |
| --- | --- | --- | --- |
| Volatile corrected/uncorrected ECC | `NVMLHealthReader`, existing `Device.TotalEccErrors` | Exact uint64 and unsupported/denied/lost fixtures | Retained frozen `62da` compiled C/Go native health receipt; exact-current-head device qualification remains a gate |
| Corrected/uncorrected remapped rows, pending/failure | Existing scalar `Device.RemappedRowsV1` | Separate fields, boolean units, per-query bounds | Retained frozen `62da` compiled C/Go native health receipt; exact-current-head device qualification remains a gate |
| Temperature, power, throttle bits | Existing NVML methods | Measured/unavailable fixtures | Exact-head physical NVML read; no thermal fault injection |
| PCIe width/generation | Existing NVML methods | Canonical generation/lanes and identity controls | Physical supported-device read |
| Selected NVLink states | Existing `Device.NvLinkState` | Selected indices, scoped boolean values | Physical link read; no implied traffic/peer identity |
| XID critical events | `NVMLXIDReader`, existing event-set API | Timeout, partial bounded drain, UUID/type refusal, sticky cleanup | Compiled ABI, safe native registration/poll/close; no artificial XID fault |
| Fabric/vendor health judgment | NVMLFabricReader uses existing version 3 query; raw mask/state/status/clique and separately labeled vendor summary | Host query fixtures, class separation, exact structure version and UUID controls | Exact-version physical receipt pending; direct DCGM and IMEX adapters described below |
| Device-to-device topology | NVMLFabricReader queries common ancestor for explicitly selected peers | Directed nominal edges, explicit validity policy, epoch, peer identity and expiry controls | Native topology receipt pending; selected remote NVLink PCI/type/version reader implemented, see [NVLINK.md](NVLINK.md) |
| CPU pressure/host memory/RDMA/network/storage/BMC | `ebpfhost/hostmetrics`, `JoinNodeHealth`, `JoinRedfishHealth` | Linux v6.8 and DMTF Sensor v1.9.0 source-derived reader/strict health roundtrip controls; exact integers, bounded input, explicit joins, inactive/unknown sensor states | Actual authenticated BMC and physical proc/sysfs/NIC/RDMA/block endpoint comparison pending |

Direct DCGM typed numeric fields now use an explicit configured device/group and
per-field status/timing; unknown or unsupported fields are unavailable. Raw
opaque vendor fields and IMEX domain/daemon status remain separately labeled
with explicit status and privacy boundaries. Governed collection refuses opaque
vendor payloads rather than treating them as numeric mapped health; these adapters do not establish
physical health, fabric connectivity, precise GPU ownership or sensor freshness.
Native DCGM/IMEX receipts remain gates.

Adjacent Linux and Redfish artifact readers consume supplied input without
opening paths or issuing requests. The optional `hostmetrics.RedfishHTTPClient`
performs caller-configured bounded requests and same-origin collection paging;
physical authenticated BMC qualification remains a separate gate. Composition requires explicit node/device and clock
identity; chassis power never becomes GPU power. Unknown sensor timestamps remain
unknown. See [hostmetrics](../../../ebpfhost/hostmetrics/README.md) for supported
source formats and [NVLINK.md](NVLINK.md) for selected-link remote PCI evidence.

The frozen `62da` compiled C/Go health comparison is a bounded historical native
receipt, including unsupported fields. It does not qualify every later candidate
or every health source.

`NVMLHealthReader` queries UUID before and after each batch and refuses handle
identity changes. The constructor queries the NVML version and stores it in
every bundle. It never enumerates global devices. Supply an initialized handle
and an independently established MIG scope, or use `MIGScopeUnknown`.
Volatile ECC values are driver-lifetime counters. No integration interval or
sensor timestamp is manufactured from query time. An actual `GPU_IS_LOST`
response downgrades the batch to unavailable rather than retaining stale
measurements. The implementation never resets errors, changes NVLink settings,
sets power, repairs rows or invokes diagnostics.

The XID reader creates and registers its own set for one UUID. It never accepts
a caller's multi-device set. `Poll` performs a bounded number of zero-timeout
waits. Empty results mean no event was observed, not that a GPU is healthy.
Raw native GPU/compute instance IDs remain uninterpreted scope strings;
per-instance event coverage is not claimed. A drain that reaches its budget
returns partial observations and an error. Append these to a bundle only with
the appropriate completeness/loss decision. Do not discard partial errors.
Failed event-set teardown retains ownership for an explicit caller retry.

Native error text is retained in raw health observations for diagnostics.
It is not a general redaction mechanism; external publication must omit
`NativeError` unless diagnostic text has been explicitly approved. The
governed sampler below suppresses metadata warnings and native error text in
metadata failures before collector storage; typed property/reason remain.

## Collector composition

Read an explicit version 1 `CapturePolicy`, create a target-scoped source,
and acquire `CaptureRegistry.NewSampler`. Pass that sampler and
`policy.Collector` to the existing `NewCollector`/`NewWriter`. The runnable
`ExampleGovernedSampler` covers policy, sample, completion and strict replay.
For physical NVML, initialize NVML and select handles before constructing
`NewNVMLSampler`; its empty-device constructor enumerates globally and is not
a target-selection profile. The governed wrapper rejects a source whose
`Devices` set differs from the exact allowlist before querying it, then checks
identity before and after each batch. A source must truthfully implement
`Devices`; this wrapper is not a privilege boundary against hostile samplers.

Policy version, target UUIDs, positive interval, explicit fail/drop behavior,
sample record/byte limits and finite collector buffer/record/byte budgets are
validated before acquisition. Invalid replacement configuration has no effect
on an existing sampler. Reconfiguration is explicit: stop/drain the old
collector and governed sampler, then acquire a new sampler. There is no silent
hot reload or automatic recovery. Each batch is serialized without a queue.
Duplicate governed capture of one GPU is refused **in that registry/process**.
This does not replace cross-process discovery, namespace/RBAC enforcement or
the native GPU lease used by hardware qualification.

`Status` reports successes, failures, in-flight state, last query duration,
last successful query and suppressed diagnostics. `Ready` fails for closed,
last-query-failed and stale/unqueried sources. A successfully returned batch
containing unavailable optional counters does not fail source readiness.
`Close(ctx)` prevents new queries and cancels the in-flight query. If drain
times out, ownership is retained until `Sample` actually returns. Sources must
honor context; Go cannot interrupt a blocked native call. Cancellation is not
device completion. Export/delivery outages and queue drops use the separate
delivery API's counters; they do not fabricate device health.

`SafeCaptureLabels` defaults to no labels. Explicitly select bounded `job_id`,
`namespace`, `executable` or `stack_symbol` identifiers. Other values, environment
and command strings, credentials and model inputs are unsupported. Rejected
labels never enter the returned storage/export map. This conservative selector
is not a general detector for secrets hidden inside a permitted identifier.
Untrusted unavailable reasons are rejected; use the typed reason constants.
Additive managed `Reading.Source` values are validated against the retained
mapping and raw numeric value, source ID and producer before capture. The
wrapper copies nested source/label storage, bounds strings before allocation,
and checks each encoded record against the batch budget without constructing
an oversized batch. GPU identity labels survive; other source labels are
suppressed unless `CapturePolicy.AllowedSourceLabels` explicitly selects a
supported safe identity label. Suppression is counted in `Status`. Sensitive
API-key/environment labels cannot be selected through this policy. Health
bundles also validate managed source mappings and producer versions; raw health
bundles are not automatically governed external publications.

## Deployment and operational gates

| Mode | Implemented composition | Qualification |
| --- | --- | --- |
| Bare-metal NVML-only | Caller initializes NVML, selects UUID-bound handles and uses governed collector; no BPF/CUPTI privileges or injection | Host fixtures; physical NVML permissions/ABI remain pending |
| Slurm | Runnable bounded NVML-only script with actual CLI UUID allowlist; explicit opt-in `job_id` labels in library | Scheduler allocation must be independently verified; real Slurm receipt pending |
| Kubernetes | Non-root, no API-token/RBAC/capability NVML Pod template with UUID selection and resource/volume caps | Operator supplies immutable image and assigned UUID; device plugin/runtime and real cluster receipt pending |
| BPF source | Existing caller-managed source/lease; governance adds no capabilities | Use BPF backend's exact-head privilege/native receipts; not NVML qualification |

There is no installer, privilege escalation, background discovery, telemetry
publication or shared deployment here. Profiles in `testdata/profiles` use
the actual CLI UUID, byte and buffer flags. Native handle UUIDs are checked
before constructing a selected sampler; these profiles do not rely on
CUDA_VISIBLE_DEVICES to filter NVML.

`Retention` provides bounded complete-stream storage in a private task-owned
directory. It strictly validates existing and new streams, refuses foreign
files, symbolic links, corrupt state and duplicate writer leases, and deletes
only validated owned stream filenames. It checks changed configuration before
deletion. Publication needs at most one additional stream's disk headroom;
write/prune/rename errors are returned. Input must remain available until Store
succeeds, because pruning may precede a failed rename. Caller-established
stopped-writer recovery releases a stale lease; reopening removes an unpublished
pending file. It does not infer process death, stop a writer, automatically
restart collection, promise power-loss durability or manage shared retention.

The bounded collector test executes 10,000 fake-clock steps and checks
writer byte/record limits, buffer capacity and reported loss. It is a host
bounded-state regression, not a production soak or measured CPU/RSS/disk/
network/overhead result. Retention tests also exercise repeated bounded writes,
exclusive leases, interrupted pending recovery, corrupt/foreign/symlink state,
configuration rollback and destination write failure. `Retention.Status` now
reports validated retained stream counts/bytes, pressure, errors and prune
counters without requiring a retention mutation. This is bounded library state,
not operating-system disk telemetry or a durability receipt. Real disk-quota ENOSPC,
power-loss recovery, rolling upgrades, scheduler/namespace denial, multi-tenant
authorization and production/comparative receipts remain separate evidence
gates. Automatic process recovery and scheduler allocation discovery remain
outside these APIs.
