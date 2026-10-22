# Remaining parity blockers

The host contracts and self-oracles in `PARITY-LEDGER.md` are implemented.
The following requirements remain open. Some implementations are present;
their outstanding device and cross-implementation gates are listed separately.

* CUDA `State.Snapshot`, `Restore`, `Save`, and `Load` now have an owner-bound
  device implementation and locked GB10 receipts, including one request-local
  and two-independent-request graph replay paths. Same-request concurrent use
  is forbidden by the State contract (a State is single-request and not safe
  for concurrent calls), so it is not a missing parity feature. Physical Metal
  qualification remains open; the adapters must not be described as
  unimplemented where those CUDA receipts apply
  (`PARITY-LEDGER.md`, 2026-10-04 State graph receipt).
* Upstream KVC now has retained, hash-pinned payloads and a matching oracle.
  CUDA import and its CLI receipt contract are implemented (`beabbfa25`,
  `2efbe3ba5`). After correcting GGUF metadata admission, the continuation gate
  imported state but failed its unchanged numerical bounds: KL 0.0909226445
  and max error 4.51786375 versus 1e-6 and 1e-4 (`b31ccc78c`). Restored logits
  matched exactly. The follow-up readback checked all 229 imported sections
  and all 43 layer counts bit-for-bit, then reproduced the same continuation
  failure (`8e819f5d2`). Unused cache tails and pooled scratch were not covered;
  matched C/Go layer dumps are the next discriminator. Metal import is
  implemented (`f83500df8`), with hardware qualification still open. Host
  inspection alone is not restoration.
* Scalar CUDA speculative verification now has a locked GB10 snapshot/restore
  receipt. Opt-in grouped integration passed all 12 prefix cases for n=2 and
  n=8, and a same-site readout mutation failed (`b0b8f9adc`). This bounds the
  grouped readout and rollback fixture, not cross-implementation speculative
  parity or production throughput; those remain unqualified.
* Synthetic CUDA qualification covers the registered Q8/IQ2/Q2 routes,
  selected-expert paths, Q4_K and MXFP4 matvec fixtures, QKV D2R, HC post,
  and bounded speculative fixtures. Production quantized dispatch and
  full-model decode/prefill parity still require the registered
  cross-implementation workload.
* Server golden qualification now has the finalized upstream G1-G9 transcript
  corpus and host comparator. The focused replay at `2021a664d` matched G3,
  G4, both G6 modes, and G7 in all ten fresh A/B processes (`986791f21`). G9
  concurrent/serial replay and a full-corpus run at one head remain required;
  normalized response equality does not establish device token identity.
* CLI sampling/thinking defaults are resolved and implemented. Upstream
  seeded-sampling and full-model agreement gates remain separate requirements.
  The older logprob observation path selects argmax and is not sampled-stream
  evidence. Actual selected-token/RNG/preselection-logit tracing is implemented
  by `61b8d94a1` and `0ebad85a7`; matched physical sampling remains open.
* Hard-smoke now has the source-extracted twelve-case corpus and a real
  State-backed execution path (`7116c5474`), with host controller tests.
  Attested upstream trace import, matched device execution, cross-regrading,
  and device-visible mutation gates remain open. A comparator alone does not
  qualify model execution.

The speculative host contract now fails closed when compensating restore fails
(`c42c49461`): verification and restore errors are joined with
`ErrDesynchronized`, and a regression test checks both causes. This closes a
host error-reporting gap, but it does not qualify the device drafter, batched
verifier, or cross-implementation speculative path.

These are explicit UNMEASURED/PARTIAL states, not completed parity claims.

The e-graph now emits `rms_quant_q8_1` (and its legacy
`ds4_rms_quant_q8` alias) and slot-aware `fused_swiglu_q8_1` CUDA forms, with
independent host oracles and mutation tests. The branch-matched Q8_1 suite now
passes under `gpu-lock` on GB10 at source `734274693`; receipt and hash are in
`PARITY-LEDGER.md`. This qualifies the synthetic suite only, not production
dispatch or full-model parity.

The isolated Q8_0 CUDA qualification is now PARTIAL: quantizer and matvec
fixtures passed on DGX GB10 under `gpu-lock`, including mutation and shape
coverage. This does not qualify IQ2_XXS/Q2_K production dispatch, the full
model, or Metal.

The IQ2/Q8_1 truncation control is also physically exercised: the DGX synthetic
kernel test confirms the registered integer-toward-zero device result (+/-5)
against the independent mathematical reference (+/-5.375). This is a
targeted semantic control, not production IQ2/Q2_K dispatch or full-model
parity.

Q8_1 activation quantization and batched block quantization are now physically
covered by synthetic DGX controls. The tests passed byte-exact fixtures and
short-input/output/stride mutation cases, but this still does not qualify the
production expert routes or full-model behavior.

The registered full-model CUDA oracle trace was also rerun on DGX. It remains
FAIL: 1 of 10 teacher-forced selections disagreed, with 4/10 steps below the
leader margin needed to trust agreement; the raw-logit comparison reported up
to 4.61 absolute error. Receipt: `~/tmp/ds4-oracle-trace-20260930.txt`.

The selected-expert Q8_1 gate previously failed at the activation byte check;
the round-to-nearest fix in `e3ddbbdf6` superseded that receipt. The current
branch-matched suite covers the selected routes and their mutation controls;
production full-model parity remains open.

The analogous Q8_K expert route initially exposed the same fast-math rounding
class. `4bf86d253` makes its reciprocal, product, and final scale use explicit
round-to-nearest intrinsics; the DGX expert qualification then passed at one,
multiple, and production input widths.

The combined expert-dispatch, gather-differential, and SIMD-vs-scalar DGX
receipt also passes, including odd-tail shapes, selected IQ2/Q2 routes,
rotated-routing negative controls, and Q8_K byte identity. Receipt:
`~/tmp/ds4-expert-dispatch-fixed-20260930.txt` (SHA-256
`54750853383d29673de80eeead6764edfc7dfc265a8b13662890462f61464f81`).

The mapped-model control produced the same full-model failure as the default
resident run: step-0 max error 4.60992, 129225/129280 violations, and step-9
choice 2052 versus oracle 1. Receipt:
`~/tmp/ds4-oracle-mapped-20260930.txt` (SHA-256
`3052f9a900cb3e68b8dd278efe5023026201699ca8bb9afb1ff8377ecb8b25e7`).
Changing mapped versus resident loading is not supported as the cause; state
localization remains required.

The host-only State persistence self-oracle now covers snapshot restore,
Save-Load-Save identity, truncated-load rejection, and failed-load atomicity;
CUDA and Metal persistence remain open. The G7 hard-smoke comparator rejects
missing terminal payloads, and the speculative identity gate rejects traces
without accepted drafts, complete margins, or an eligible pre-near-tie
position. These are host contract gates, not physical backend qualification.

The DGX deep-context decode self-qualification passes: 2048-token prefill ends
with ratio-4 attention/indexer counts at 512, sixteen decode steps advance them
to 516, and all measured phases retain positive residue. Median full-step time
is 71.29 ms. Receipt: `~/tmp/ds4-deep-decode-20260930.txt` (SHA-256
`21382db3ad2af3eb49ad892ad70dc371f78f552f3b01071272535f373cce2a47`). This
does not establish C-vs-Go decode parity.

The oracle trace was rerun with reusable Go state output. It still fails with
one of ten selected-token disagreements, step-0 maximum error 4.60992 and
129225/129280 violations, and step-9 selection 2052 versus oracle 1. The
state directory is `~/tmp/ds4-oracle-go-states-20260930`; its manifest SHA-256
is `106be6117ab3dac6138adec0afcdb21dea19c1ea71d6cb1c0e16fb8783f4d31f` and
the receipt SHA-256 is
`bc30100fb3d725d1a6b374902285e4f460baae10b9d44d7864b21c95c836622c`.
This leaves terminal-position and first-divergent-layer localization open.

Host progress: CLI sampling defaults are now sourced from the sampler package
(`31cb623fe`), and server transcript comparison has strict content/token
checks with volatile identity masking (`607751bbe`). The finalized upstream
G1-G9 corpus is now host-loaded and set-compared (`3e6a03473`, `2eb4b5ba5`);
physical CUDA/Metal state persistence remains unmeasured.

Server requests now inherit upstream sampling defaults when omitted, with
explicit zero temperature retaining greedy behavior (`6f4a28b76`,
`2ccfaa0eb`). This closes the host default-policy drift; the DGX G1-G9 replay
exists but remains unqualified because response and concurrency contracts differ.

The runtime server entrypoint is now present as a narrow Linux-only `-server`
mode in `package main` (`1bcf1a9a4`). It reuses the existing model loader,
tokenizer, CUDA runtime, and `dsv4.State` adapter; each HTTP request receives a
fresh request state and the process owns the backend for its lifetime. A
non-Linux build refuses the mode explicitly, and focused host tests plus Linux
vet pass. This removes the former architectural entrypoint blocker without
claiming that the server has been run against the upstream corpus.

The NVRTC signal-disposition failure found during server qualification is now
guarded at the compile site (`1692d4924`). Linux `nvrtcx.Program.Compile`
serializes compilation and restores the complete SIGINT/SIGTERM/SIGXFSZ
`rt_sigaction` records, including Go's `SA_SIGINFO` and alternate-stack flags;
the synthetic exact-restoration test and package checks pass. This removes the
known compile-induced SIGTERM crash from the source path. Commit `e2100a99e`
restores graceful SIGINT/SIGTERM shutdown on top of the guard. The DGX replay
receipt exists, but it remains unqualified because the response and concurrency
contracts still differ.

The remaining server hold is runtime qualification: the branch-matched server
has now been run on DGX against the finalized G1-G9 corpus, but the receipt is
not qualified. Response contract, tool-history, concurrent graph isolation, and
token-identity gaps remain; see the dated replay entry below.

Quantized expert dispatch now rejects unsupported Q4_K/MXFP4 and malformed
layouts before allocation, while admitting the supported IQ2_XXS/Q2_K CUDA and
Metal shapes (`b4c316d10`). This does not qualify the physical kernels or full
model across quant variants.

Phase 0 configuration admission now validates dimensions, clamps, and layer
routes before backend use (`2756c02d9`). Unknown compression ratios fail
closed instead of selecting a coarse route. CUDA and Metal State snapshot,
restore, save, and load are now implemented for the owner-bound CUDA State
adapter with locked GB10 receipts. Metal hardware persistence, same-request
graph concurrency, and broader cross-implementation qualification remain
open.

The G7 comparator also regrades each terminal output against the local case
table and rejects mismatched `status` or `picked` fields. This prevents two
traces from passing merely because they repeat fabricated producer metadata.
The finalized upstream G1-G9 corpus is loaded and compared by the host
contract (`3e6a03473`, `2eb4b5ba5`); physical cross-implementation G7 execution
remains unmeasured.

The KV store now has a bounded `InspectUpstreamFile` path for seekable files.
It validates the upstream KVC header and declared body sizes using the caller's
file size without reading opaque payload bytes, and reports trailer size for
inventory tooling (`262163ab3`). Payload restoration remains explicitly
unsupported; CUDA/Metal State persistence and upstream payload interop are
still unmeasured.

The finalized upstream KVC sample was re-audited against the current section
map on 2026-10-05. Host parsing accounts for every payload section, but restore
still needs a version-qualified CUDA owner API, model/token provenance, graph
invalidation, atomic rollback, and a fresh-process continuation receipt. The
private DS4CSTAT loader must not autodetect the upstream envelope, and Metal
precision prevents treating a CUDA-only experiment as full feature parity.
Receipt: `/Users/tmc/tmp/ds4-lib-workflow/20261005-kvc-restore/REPORT.md`.

The benchmark package now exposes a paired `Compare` helper that aligns rows by
context and token workload and reports base, target, and paired deltas for
prefill and decode metrics (`4f24d747d`). It is an analysis primitive, not a
qualification verdict: warmup, interleaving, host-condition, and correctness
receipts remain required.

`cmd/ds4-bench` now exposes that comparison as a strict JSON mode with paired
CSV inputs (`667f2b39a`). It refuses stdin comparison paths and conflicting
single-table flags; it still does not run a model or qualify a benchmark.

The exact-sampling host gate now exercises every possible draft token, not just
one fixed draft, and rejects an always-accept mutation (`b107fec58`). This
strengthens the distribution contract; speculative device snapshots, verifier
integration, and accepted-draft runtime qualification remain open.

The KV inspector also has a path-based entry point that stats and bounds a
regular file before reading its fixed metadata (`9461d1f56`). Upstream payload
restoration remains deliberately unsupported.
The next server runtime slice remains separate from the host contract. The
local `dsv4server` loader/set comparator and mutation tests are complete, and
the finalized upstream G1-G9 corpus has been replayed on DGX. Its measured but
unqualified receipt is retained at
`/home/tmc/tmp/ds4-agent-kvc/20261005-server-runtime/rcb37/`.

The host-side G5 tool-history boundary fix is now in `acf5bbf96`: an empty
leading `</think>` marker in a tool request is treated as ordinary final
content, while strict marker validation remains in the ordinary thinking
paths. Its DGX replay is pending, so this narrows the source change but does
not change the runtime qualification status.

A source audit found no safe host-side repair for the measured G9 concurrent
graph desynchronization. `dsv4server` allocates independent sessions with
FIFO/cancellation-safe pooling, and the state contract forbids sharing one
request state concurrently. Serializing or reducing `SessionSlots` would hide
the failure rather than repair it; the remaining fix is CUDA graph/runtime
state work, with a fresh device receipt required.

Speculative host coverage now checks every accepted-prefix length, including
zero-accept snapshot restoration and rejected-draft retention
(`8e344cd17b8cc174cd60f5fd756bce23f027b782`). This strengthens the host
contract but does not qualify device snapshot/restore or runtime speculative
decoding.

The 2026-10-05 speculative audit found no additional safe host slice. The
remaining G1-G3 work is device-owned: DSpark/MTP drafter loading, fused or
batched verifier parity, cross-implementation token identity, and production
device gates. Receipt: `/Users/tmc/tmp/ds4-lib-workflow/20261005-speculative-audit/RECEIPT.md`.

The Phase 2 G1-G3 audit found no remaining host-only gap: session persistence,
split invariance, malformed/truncated input, and failed-load atomicity are
covered. CUDA/Metal persistence remains HOLD because device synchronization,
copy, graph invalidation, and runtime receipts are collision-owned.
Receipt: `~/tmp/agent-collab/ds4-min/20260930-phase2-g1-g3-audit-hold.md`.

The Metal State adapter still fails closed with `dsv4.ErrUnimplemented`, and
its refusal/non-aliasing/close tests remain host-scoped. CUDA State now has
owner-bound snapshot/restore/save/load paths and the single-request graph
receipt above; concurrent graph replay, Metal persistence, and broader
cross-implementation runtime behavior remain HOLD pending their respective
device payload, synchronization, and graph receipts.
Receipt: `~/tmp/agent-collab/ds4-min/20260930-device-state-adapter-audit-hold.md`.

Read-only upstream KVC loading now has a bounded host API
(`dsv4kvstore.ReadUpstream` and `ReadUpstreamPath`, `0f3cf2a2e`) and records a
SHA-256 fingerprint for the opaque payload (`a696339ce`). It validates the
envelope before allocation and returns text, opaque payload, and trailer bytes.
Backend restoration and cross-implementation logits remain UNMEASURED.

### 2026-10-05 server corpus replay: MEASURED, NOT QUALIFIED

The finalized upstream G1-G9 corpus was replayed on DGX from `cb37e93c5` under
`gpu-lock`. Binary SHA-256: `1ebbc5f5b6a3be9a85aa24702dae2a317ced76490a841d1259d4dd488c188b40`;
`vcs.modified=false`; 32/32 server processes exited normally with zero SIGSEGVs.
Receipt: `/home/tmc/tmp/ds4-agent-kvc/20261005-server-runtime/rcb37/`.
Only the models listing and malformed-JSON cases matched exactly. Generated
content/reasoning, SSE framing, usage fields, thinking-disabled behavior, and
incomplete output still differ. Tool-call history returns 500, assistant/tool
extension history returns 400, and concurrent G9 slots expose CUDA graph state
desynchronization while serial controls remain repeatable. Token identity is
UNREAD because the HTTP surface does not expose token IDs; runtime parity remains
open.
