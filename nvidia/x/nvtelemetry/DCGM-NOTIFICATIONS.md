# Read-only DCGM policy notifications

The existing cached field sampler is separate from policy notification callbacks.
`dcgm.NewPolicyNotifications` registers only `dcgmPolicyRegister_v2` and owns the
matching unregister operation. It does not set or trigger policy, remediate,
start/stop an engine, or call global shutdown. No generated binding is changed.
The response ABI is pinned to NVIDIA/DCGM commit
[64df9f894541e426e416131a9820cae97aa4dd81](https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/dcgmlib/dcgm_structs.h):
v2 stamp 0x2000020, 32-byte response, condition union and GPU index.

Supply an independently owned private Session group containing only selected
GPUs. Process-local overlap refusal cannot establish isolation from external
registrations; borrowed or externally shared groups are outside this contract.
The caller must retain its group/connection until successful subscription Close.
A native registration error returns a non-nil owner with the error because
native side effects cannot be presumed absent. Explicitly Close that owner;
a failed unregister keeps its callback and ownership active for caller retry.
Missing symbols refuse before native registration. Legacy registration is not
aliased to v2.

One permanent purego trampoline receives a typed native response pointer and a
numeric never-reused token. No Go state or host-buffer address is encoded as a
uintptr. Only the four-byte version header is inspected before the known v2
payload is copied. Short legacy/unknown layouts, nil pointers, queue overflow
and callback lock contention increment loss; no unknown union is dereferenced.
Tokens retire only after successful unregister and synchronize channel close.
Late callbacks hit the permanent trampoline and are ignored. This is Go-side
queue safety, not proof of native callback quiescence or permission to unload
DCGM while native workers may still execute.

`NewDCGMPolicySource` consumes these copied callbacks with cancellation, bounded
caller-supplied inventory and source/clock lineage. `WriteDCGMPolicyEvent` writes
one bounded JSON diagnostic line. uint64 union words remain exact, device/layout
unknowns remain explicit, and consumer arrival time is never called sensor time.
No opaque condition union is promoted into a metric or health judgment. These
raw events are not OTLP event/log or normalized health conversion.
`loss_unknown` is always true: counted helper drops do not establish upstream
coverage. Clock identity must match supplied ClockMetadata and remain stable
through arrival capture; invalid UTF-8 identities are refused instead of repaired.

Host controls compile an independent C response mirror and run actual native
thread callbacks, including a four-byte header against a protected guard page,
queue loss, register/unregister failures, late callback retirement and library
reload. Adapter controls decode its public JSON output independently. Real
DCGM notifications, driver/device permissions, selected-group policy state,
physical trigger evidence and native shutdown/quiescence remain admission gates.

`dcgm.ReadPolicies` optionally reads bounded v1 policy slots without setting or
triggering policy. Its typed host-buffer pointer is initialized with the 144-byte
v1 layout and checked on return; requested slot count is not measured group size.
PolicyBool/PolicyNumber setter-parameter constructors are not needed here.
The diagnostic ErrorCode catalog separately retains the official DCGM 4.6.1
137 names/134 values and unknown uint32 fallback. It was reproduced byte-for-byte
from clean libgen 038f8646e14b86fb8eff9b697f89108c77a12ce7 against its pinned
headers, with independent compiler-value and deterministic-output controls.
These labels do not turn diagnostic errors into remediation or health judgment.
