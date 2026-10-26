# Passive health-capacity command fixture

`nvsentinel-v1.26-request.json` is constructed from the public shipped NVSentinel
v1.26.0 schema. It is not an authentic staging capture, an authorized request,
a deployed test result or evidence of physical GPU validation.

Fixture SHA256: `56e5702920e4ca0717c6d2cc886cc61f4b4bf3157af4c489d785435f46fcca87`.
The manually selected request asks for node-a/node-b and dcgm-diag-test/bandwidth.
Its producer-reported Succeeded status retains a failed TestTimeout attempt on
node-a, a later successful attempt, and skipped node-b/bandwidth. The importer
must retain all of those states and withhold coverage and independent proof.
Provider objectName values are references; they supply no raw output, GVK,
namespace/UID identity, execution image hash, authorization or device binding.

## Immutable schema sources

Release v1.26.0 resolves to commit
`a5b9e90fb3729d72f9ebc1bddcbf4eb518c0b3f5`. These official source files were
retained and hash-checked before this implementation; no cluster or download is
needed to use the command or run its host controls.

- [ValidationRequest types](https://github.com/NVIDIA/NVSentinel/blob/a5b9e90fb3729d72f9ebc1bddcbf4eb518c0b3f5/lifecycle-manager/api/v1alpha1/validationrequest_types.go):
  Git blob `dbd1e41fee5907aa56eb083572ec010ef113c0ed`, SHA256
  `2318446161532306d4df361e1bcc433bc593f0bc07ee95ffb3961bf3f9853ca7`.
- [Installed CRD](https://github.com/NVIDIA/NVSentinel/blob/a5b9e90fb3729d72f9ebc1bddcbf4eb518c0b3f5/distros/kubernetes/nvsentinel/charts/lifecycle-manager/crds/nvsentinel.nvidia.com_validationrequests.yaml):
  Git blob `428e72e9115656a4cf0f09310dbaa8a511f8684f`, SHA256
  `e677e24cdba08a8b057884150f76f9b2c64cb6c9685a549ff0623a8c012c3548`.
- [Group/version](https://github.com/NVIDIA/NVSentinel/blob/a5b9e90fb3729d72f9ebc1bddcbf4eb518c0b3f5/lifecycle-manager/api/v1alpha1/groupversion_info.go):
  Git blob `e3e22f29ee8595830585e0dd301d60282e9d6f34`, SHA256
  `4a3ce0adebe85f6ed94338295a7584adb0254c46eab0245ffa9d523e90582a56`.

The wire selection is `nvsentinel.nvidia.com/v1alpha1`, `ValidationRequest`.
NodeSpec has name only. Proposed ADR-only GPUUUID/exclusiveAccess fields are
refused. Desired tests default through active configuration when empty; without
that separate configuration the importer reports defaults_unresolved.

## Local command

```
cudaprof health-capacity -health health.json -telemetry telemetry.jsonl \
  -input options.json -nvsentinel nvsentinel-v1.26-request.json \
  -nvsentinel-sha256 56e5702920e4ca0717c6d2cc886cc61f4b4bf3157af4c489d785435f46fcca87
```

`-nvsentinel` and its expected raw SHA256 are optional. Digest matching proves
which bytes were read; it does not authenticate their source. Every input is an
explicit regular local file. The command does not read stdin, fetch URLs, query
a cluster, execute providers or request validation/recovery actions.

`options.json` uses command fields `telemetry`, `health_limits`, `health_findings`,
`work` and `limits`, containing the existing TelemetryOptions, HealthLimits,
Finding, AccountingWork and HealthCapacityLimits JSON representations. Their
existing exported field names/tags apply. TelemetryOptions is decoded separately
and explicitly assigned because HealthCapacityInput.Telemetry has json:"-".
Receipt gates and other supplied admission evidence remain caller assertions;
the command introduces no independent authority verification.

The existing public ComposeHealthCapacity computes the incident. The optional
import adds only opaque exact-byte request/status refs and a separate parsed
producer wrapper. It cannot change health/workload association or Impact.
Missing status, empty status, all four phases, retries, failure details,
membership, skips, time fields and conditions are retained. Physical coverage,
origin, authorization, node/device association, active configuration, raw
provider objects and numerical outputs remain unverified or unavailable.

Input files default to 8 MiB each (`-max-bytes`, hard cap 64 MiB); the producer
request has an additional 8 MiB cap. JSON limits are 32 nested levels, 32768 bytes
per string/key, 4096 entries per array and 200000 tokens. The producer decoder
also limits 1024 node/test members, 256 groups, 64 attempts per group/4096 total
and 128 conditions. Existing library limits apply independently. Output defaults
to 8 MiB (`-output-bytes`, hard cap 64 MiB), checked before writing. Malformed,
duplicate, trailing, incompatible, oversized or inconsistent time/state input
returns an error without structured output.
