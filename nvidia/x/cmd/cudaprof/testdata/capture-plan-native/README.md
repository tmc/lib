# Cooperative native capture-plan fixture

This Linux target prepares one real NVRTC CUBIN and loads those exact bytes
in each requested process. It emits no synthetic CUPTI counters. It requires
an inherited `gpu-lock` for preparation and collection. Native execution has
not been performed for this fixture.

From `nvidia/`, compile for the intended host, then on the authorized GPU:

    go build -o <task-dir>/capture-plan-native ./x/cmd/cudaprof/testdata/capture-plan-native
    gpu-lock <task-dir>/capture-plan-native -prepare -dir <packet-dir>
    gpu-lock cudaprof capture --capture-plan <packet-dir>/plan.json --capture-plan-cooperative -- <task-dir>/capture-plan-native -dir <packet-dir>

Confirm the current CLI's capture-plan flag spelling and normal injection,
output, and source mapping options before a native run. The parent runner
retains all commands, provider receipts, and source/PTX/CUBIN/SASS files.
Do not infer qualification from this command template.

The default plan has activity, range-replay, and sass-replay passes with distinct
replay identifiers. The same 64-thread workload overwrites every output with
4096 uint32 LCG steps. Every output is independently checked before publication.
Range attribution is one named `work` region using `sm__inst_executed.sum`.
Preparation discovers the actual SASS MetricID; GPU granularity is requested.
Unsupported native features and unknown loss remain incomplete. This fixture
does not replace lane 60's independent tutorial comparisons or isolation tests.

PM is optional: preparation accepts `-pm-config <actual-config.json>` containing
an independently constructed `cuptipmsampling.Config`, including its real PM
configuration image, metric names, sampling interval, and trigger/append modes.
There is no PM configuration-image builder in this owner package. A range
configuration is never substituted. PM sampling runs the checked workload in
the same target primary context but remains device-wide; the owner exposes no
Offered sample denominator, so the result remains partial even when decoded.

Preparation writes `packet.json`, `plan.json`, `kernel.cu`, `kernel.ptx`,
`kernel.cubin`, and `kernel.sass`. The packet binds the compiled artifact,
NVRTC cache key, exact source, native disassembly, and workload. It does not
attest that another process loaded it. The target loads the recorded CUBIN;
the controller must independently admit its injected module journal and exact
loaded image. Missing code association remains partial. Source/code packet
retention in the final archive is the runner's responsibility.

Observed device name, driver API version, loaded NVRTC version, CUPTI version,
and current SM/memory clocks are queried again in the target. `CUDA` is labeled
`NVRTC major.minor`, not represented as a CUDA runtime version. Current MHz is
an observation, not a locked-clock guarantee; a changed clock receipt can cause
honest plan-environment mismatch. The random fixture correlation domain is an
explicit experiment label; it establishes no cross-pass timing causality.
CUPTI supplies timestamps and context ID. Linux `/proc/self/stat` supplies the
actual process-start generation; this is producer provenance, not a guessed
injected-journal generation. Activity and PC identities remain controller-owned.

Controller-provided positive limits are mandatory:
`CUDAPROF_CAPTURE_MAX_RECORDS`, `CUDAPROF_CAPTURE_MAX_BYTES`,
`CUDAPROF_CAPTURE_MAX_PASSES`, and
`CUDAPROF_CAPTURE_DEADLINE_UNIX_NANOS`. The fixture clamps them to admitted
plan limits and applies them before owner collection and every workload launch.
Range native starts use the supplied remaining budget. SASS FlushBounded checks
one properties observation against aggregate raw/decoded/header/name storage
before allocation. PM sample/hardware buffers and the native-requested counter
image are bounded before allocation. The fixed executed workload descriptor
is checked exactly; a PM configuration for another device is refused. Pure
budget and workload controls run without Linux or a GPU. Tool output, input
packets, and publication bytes are bounded.
CUPTI calls are synchronous and cannot be canceled mid-call by this fixture;
the controller owns the independent process deadline. Session cleanup errors
are included in incomplete collection receipts. Device/module/output teardown
errors fail the target rather than pretending cleanup succeeded.

Publications are written only into the actual `CUDAPROF_TRANSPORT` directory
provided by the controller. No target journal IDs or expected environment fields
are copied into observations. Missing provider, dropped counter, and stale
artifact controls belong to the parent qualification campaign; this fixture
alone supplies no claim that those controls or native acceptance passed.

`-pm-metric <name>` is an alternative to an external PM configuration file.
It constructs a distinct PM host object using the existing CUPTI bindings,
restricted to the measured CUPTI 130400 initialization layout. Configuration
bytes come from that PM object, never a range object. Native required passes
must equal one. Requested metrics unsupported by the device produce
`pm-unavailable.txt`; no PM request with invented configuration is added.
The sampling policy is explicitly 1000 ns, time-interval trigger, keep-oldest,
128 sample capacity and 1 MiB hardware buffer, all further clamped by runtime
budgets. It is unqualified until the authorized native run records outcomes.

Primary API reference: [NVIDIA CUPTI PM sampling usage](https://docs.nvidia.com/cupti/main/main.html#cupti-pm-sampling-api).
Host test doubles require PM profiler type, reject multipass/unknown-layout/
oversized output, and retain pinned ownership after failed native close.

If SASS metric discovery fails, preparation writes `sass-unavailable.txt` and
uses a second distinct range replay, without manufacturing a SASS MetricID.
Each range/PM/SASS publication now has a standard code-only
`capture-provider-<provider>.pprof` sidecar, using `cudacapture.WriteLimits`.
It retains source/PTX/the exact loaded CUBIN/native SASS; its profile contains
no synthetic timing samples. Its producer module identity names the loaded
handle, not an invented CUPTI module ID. This is a producer receipt, not driver
attestation. Checked actual transport sizes and nested base64 encoding sizes
bound publication and code packet bytes before writing. The controller
separately checks later journal growth and final manifest encoding. Range
counter storage reserves three quarters of remaining bytes for retained and
encoded copies; the original plan limits are not increased.
