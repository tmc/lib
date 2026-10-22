# Upstream CUDA checkpoint continuation gate

Production import is explicit:

```
ds4-min -backend cuda -model MODEL -ctx 4096 \
  -kv-load COLD.kv -kv-load-contract CONTRACT.json \
  -prompt-tokens COLD.tokens.json -tokens 1
```

The import receipt has `ModelSHA256`, `TokenIDsSHA256`, `PayloadSHA256` hex
strings and the full `Contract` geometry object described below. It is distinct
from the continuation gate receipt (which also names a second checkpoint and
comparison bounds). The loader validates the complete payload/token provenance
and updates the session timeline only after the backend State upload succeeds.
Unsupported backends reject it; ordinary local `-kv-load` is unchanged. Passing
the exact stored token IDs as the prompt preserves the imported frontier.
Model identity and transport safety do not alone establish numeric parity.

`TestCUDAUpstreamContinuation` is an opt-in fresh-process restore gate, not a
default `-kv-load` fallback. A KVC does not identify its model: supplying only
the filename cannot authorize a device upload.

Build the Linux test binary from a clean revision, record its SHA-256 and VCS
revision, and invoke exactly this test under `gpu-lock`. Set
`DS4_GGUF_PROFILE_MODEL` to the model and `DS4_UPSTREAM_RESTORE_MANIFEST` to a
JSON receipt prepared independently of the files being restored. The test
refuses unknown JSON fields, trailing data, missing hashes, geometry mismatches,
and missing/zero/nonfinite agreement bounds.

The receipt has these fields:

- `RecordPath`: cold upstream checkpoint.
- `ContinuationPath`: checkpoint after exactly one teacher-forced token.
- `ModelSHA256`, `TokenIDsSHA256`, `PayloadSHA256`,
  `ContinuationPayloadSHA256`: full hex SHA-256 from the upstream receipt.
- `Contract`: `dsv4kvstore.UpstreamRestoreContract`, with every geometry field
  and `Ratios` pinned to the model/source ABI. Its hash array fields can be
  omitted: the test fills them from the explicit hex fields above.
- `MaxKL`, `MaxAbs`: finite positive pre-registered continuation bounds,
  qualified on the reference control for this fixture. There are no defaults;
  do not import a tolerance from a different prompt or shape without a control.

The 2026-10-04 sample's cold/shutdown pair contains 2057/2058 tokens. The cold
record is not the bare 2048-token workload: its chat template shifts all model
positions by seven. Use the sample receipt and full token hash, not the raw
workload hash. The continuation token is 1300; the gate recovers it from the
independently fingerprinted shutdown record and verifies the entire prefix.

Before decode, the gate requires exact restored host and device logits and
tests rejection of a mutated token-provenance hash without changing the live
frontier or logits. It then reports `KL(C || Go)` and max-absolute logit error
against the continuation record. A numerical failure is a failed continuation
gate, even when device upload and malformed-input controls pass.

The codec stages into a separate owned snapshot, clears unused cache tails and
pooling scratch, maps logical raw rows into the target ring, copies compressor
partial-window state, synchronizes and invalidates graphs, then commits. A
failed live copy rolls back from an unchanged backup. A failed rollback marks
the State desynchronized. Synthetic tests cover rollback decisions; a physical
copy-fault/rollback test and the fresh-process continuation run still need
their own hardware receipts. CPU tests and cross-compilation do not close them.

## 2026-10-07 physical continuation receipt

Clean revision `5b80b6c154e5109d7b26fc7f8fdebd0d3477bf33`; Linux test binary
SHA-256 `232250640ddf45f61b0842fb3ff313bbf7723e7e66ecefb270e6a35880aa2829`.
The corrected run acquired and released gpu-lock label
`D2546E0A-kvc-restore-corrected`, rc 1, after 184.27 seconds.
Exact restored host/device logits and malformed-receipt frontier preservation
passed. Teacher-forcing token 1300 at position 2057 failed continuation agreement:
KL 0.0909226445 and max absolute 4.51786375, against the original pre-registered
HC-control bounds 1e-6 and 1e-4. No bound was relaxed and no retry followed.

The independent C fused/unfused HC controls changed the executed kernel census
but reproduced the exact retained shutdown payload in all four processes.
Their zero spread activates the pre-registered floors; it is not evidence for
a general numerical tolerance. A separately registered QKV control also had
zero spread and does not replace the HC rule.

Raw log and continuation logits are retained at
`~/tmp/ds4-kvc/20261007-restore/{corrected-gate.log,corrected-continuation.f32}`
on DGX and mirrored in `~/tmp/ds4-lib-workflow/20261007-kvc-restore` on the Mac.
The logit SHA-256 is
`da84dcb26a141578a10ff93e43453a39c4a9b0aeb5fff3b48f8318bacc787433`.
Device upload is measured; continuation parity is not closed. The next
discriminator is full imported-state readback before consumption, followed by
matched per-layer continuation states if the import itself is byte-identical.

The follow-up at signed revision `ff5ae72a428b9b0f9780eb54d6d53798bad15314`
used clean binary SHA-256
`2adbdaa6a8091070b3efd63dbcfd5a1e0e2940648924e1fdf1e27d908e85ac7c`.
CPU bit/count/census mutation tests passed. Under gpu-lock
`D2546E0A-kvc-state-readback`, all 229 imported raw/compressed/indexer/partial
state sections matched the upstream payload bit-for-bit before consumption.
All layer counts matched; each layer's raw capacity was checked, and the
logical-position-to-slot mapping was independently required to equal position
modulo capacity. Missing/duplicate sections are rejected. Unused cache tails
and pooling scratch were not read back, so no transport claim covers them.

The continuation again failed unchanged bounds, with identical KL/max error
and identical raw-logit hash to the previous run. Runtime 172.29 seconds,
released rc 1. There is no first failing imported section. This narrows the
next diagnostic to consumption or state outside the serialized sections; it
does not establish a particular kernel cause. No numerical retry followed.
Raw `readback-gate.log` and `readback-continuation.f32` are retained alongside
the previous artifacts on both hosts.
