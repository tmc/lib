# Metal scope-A selected Q4 experts

Implementation: signed `90a7f7489e5896fdf23a2f1ff4c27c478b020c1c`.
The exact final-six-layer splice has a source-level F32 selected gate/up/down
route, lazy request-owned pipeline resources and per-slot down inputs. Q2
dispatch and the registered-model whitelist are unchanged. No Q4 model is
admitted without its independently qualified manifest.

## Bounded physical fixture

`TestMetalQ4SelectedDeviceGate` passed on local Apple Metal, 2026-10-07,
8.60 seconds, rc 0, under `/Users/tmc/ml-explore/bin/gpu-lock`, label
`D2546E0A-metal-q4-selected`, `GPU_LOCK_MAXHOLD=2147483647`.
Memory admission before queuing: 128 GiB physical, 83% system free.
No model load occurred.

- Binary SHA256: `7df36e58d84579deca3b3f4289633d510038b1e2ac1d4010934b9971936da4c9`.
- Raw log SHA256: `bdaaa53b59a9bc4589e8e6f0751a98276b4ad55a432a2b2abee919e0ea1f0355`.
- Raw log: `~/tmp/ds4-lib-workflow/20261007-speculative-batch/metal-q4-device.log`.

The fixture uses 512 columns, 7 rows, 8 experts and 6 nonsequential selected
slots, including a repeated expert. Common F32 gate/up input and independent
per-slot F32 down input each compare 42 output values against independently
unpacked float64 dots, with abs1e-4+rel1e-4 and an explicit finite check.
This records a threshold pass, not an unreported maximum error.

A uniquely located high-scale-bit expression is changed in the compiled
kernel. Both mutant routes differ from their own unmutated outputs. Four
truncated-output cases fail before encoding; an invalid selected expert ID
writes NaN in its seven output rows rather than retaining old output. The
oracle's finite check rejects NaN; this is not a synchronous production error
return from the device kernel.

CPU tests cover packed high bits, the exact L37 splice boundary, reversed
down axes, backend dispatch and the diagnostic/gate environment inventory.
Darwin and Linux arm64 test binaries compile.

## Limits

This qualifies only the bounded Q4 unpack, selected-expert addressing,
common/slotted input and negative-control contract. Full mixed-model
continuation, other shapes, prefill/decode throughput and upstream arithmetic
order remain UNREAD. The available local CMetal snapshot unpack matches the
reference structurally but carries no Git metadata; it is not called a pinned
upstream revision. Its moe.metal SHA256 is
`17b98c548dc85e6901ffb6fec4b994bc94e3ac204de506cc96baf0110db29802`.
