# ds4-min manual gates

Evidence belongs under `~/tmp/ds4-lib-workflow/evidence/<commit>/<host>/`, not
in git. Preserve the command, stdout, stderr, and true exit status separately.

## Identity gate

Before running inference, record the canonical model path, byte length,
SHA-256, relevant GGUF metadata digest, tensor name/type/shape manifest digest,
tokenizer digest, ds4 commit and binary SHA-256, compiler, prompt SHA-256, flags,
OS, backend, device name, driver, and compute capability where applicable.
Reject any missing or mismatched model field.

## Oracle gate

Build only from a clean archive or detached worktree at `b0a147a`. For the
checked parity prompt, capture prompt token IDs; post-prefill logits; logits and
top-20 IDs/log probabilities after decode steps 1, 2, 4, 8, and 16; all greedy
IDs; the stop result; decoded bytes; timings; and exit status. Never invoke the
conflicted ds4 worktree.

## Go gates

From `examples`:

```
GOWORK=off go test -count=1 -run TestOracleGates -v ./ds4-min
GOWORK=off go test -count=1 ./ds4-min
GOWORK=off go test -race -count=1 ./ds4-min
GOWORK=off go vet ./ds4-min
GOWORK=off go test -run TestFixtureRegeneration -count=2 ./ds4-min
```

When the corresponding dependency surface changes, run its module-local tests
with `GOWORK=off` as listed in the approved workflow.

A passing `go test ./ds4-min` is not evidence of parity: almost every oracle
comparison skips unless its environment gate names a model or a captured ds4
fixture. `TestOracleGates -v` lists which gates a run exercises and what each
unset one would enable. Setting `DS4_REQUIRE_GATES` turns an unset gate into a
failure, which is how a full parity gate asserts its own coverage.

`TestCUDAPrefillGather` requires `DS4_CUDA_PREFILL_GATHER_GATE=1`, the GPU
lock, `DS4_MODEL`, and `DS4_CUDA_PREFILL_TOKENS` naming a retained 2048-token
JSON corpus. It compares gather and per-token projections across prefix
lengths and two continuation steps, checking logits and every layer's
compressed-row counts. `DS4_CUDA_PREFILL_GATHER_RECEIPT` optionally names the
snapshot output file; it is written only after every case succeeds. This
comparison does not measure performance. `DS4_MOE_GATHER_TILE` and
`DS4_MOE_MAX_EXPERTS` are CUDA compile-time macros checked against Go constants,
not environment variables or hardware gates.

## Physical gates

Run the same immutable model and prompt manifest through ds4 Metal, ds4 CUDA,
Go Metal, and Go CUDA. A receipt must report nonzero GPU command-buffer or event
time, or mark the counter unavailable with a reason; it must never encode an
unavailable counter as zero. Compare fixture intermediates at preregistered
f32 `atol=1e-4, rtol=1e-4`. Full-model results require identical greedy IDs at
each recorded step; record adjacent-logit margins rather than widening a
tolerance after observing a mismatch.

Metal additionally checks repeated open/run/close for resource leaks. CUDA
keeps context creation, allocation, launch, copies, and cleanup on the owning
context and records device name and compute capability. A compile, skip, device
count, or successful kernel compilation is not a physical pass.

Use `PASS`, `PARTIAL`, `UNMEASURED`, `FAIL`, or `BLOCKED` exactly. This section
previously said the identity prerequisite was `BLOCKED` because the pinned GGUF
was not present locally, and that full-model, Metal, and CUDA execution were
therefore `UNMEASURED`. The file is now present and its identities are pinned in
`gguf.go`, so the identity prerequisite is met; see DESIGN.md, Pinned
identities. The execution gates still require their own physical receipts.

## Comparing logit captures

A selection mismatch says two argmaxes differ. It does not say by how
much, in which direction, or whether two backends miss the reference the
same way. The logit captures answer that, and comparing them needs no
model:

    E=../../../evidence
    GOWORK=off DS4_ORACLE_STATE_COMPARE="\
    metal=$E/metal-prompt-a-states,\
    cuda=$E/oracle-provenance/candidate-prompt-a-01/candidate-prompt-a-01/session,\
    oracle=$E/oracle-provenance/capture-prompt-a-01/capture-prompt-a-01" \
      go test -count=1 -run TestOracleStateDivergence -v ./ds4-min

With three captures the last is the reference, and the correlation line
says whether the other two deviate from it together. Weak correlation
means separate defects; the two backends of this port currently sit at
+0.10 to +0.28 over the first three steps.

To produce a capture from this port, run the trace with a destination:

    DS4_GGUF_PROFILE_MODEL=$MODEL \
    DS4_ORACLE_STATE_DIR=$E/oracle-provenance/capture-prompt-a-01/capture-prompt-a-01 \
    DS4_ORACLE_STATE_OUT=$E/metal-prompt-a-states \
      GOWORK=off go test -count=1 -timeout 30m -run TestMetalOracleTrace -v ./ds4-min

Two Metal runs of that command produce bit-identical captures, so any
difference the comparison reports is a change in the arithmetic, not
run-to-run variation. That makes it a regression gate on kernel edits at
a resolution far below what a token trace can see: a change that moves
the logits but not the selections is invisible to
TestMetalRuntimePromptAGreedyTokens and obvious here.

The final-acceptance contract now requires every finite full-vocabulary
logit to satisfy `1e-4 + 1e-4*abs(reference)`. The comparison reports the
descriptive statistics above and fails when any element exceeds that
pre-registered bound. Exact repeatability between separately opened
runtimes remains a distinct byte-identity gate; failure of that gate alone
does not prove numerical non-equivalence.

## Bounded Metal raw-attention capacity: qualified

Signed `e999db47fb8b6a09b052dd8ef62a9ed58be3339f` passed one model-free
gpu-lock block on Apple M4 Max (40 GPU cores), macOS 26.6.2/25G83, Metal 4,
Go 1.27.1. Binary SHA-256:
`038b3ac9ee9655411af762b8b96b8a73465883e8ff99c39c642a9170b56397c3`.

All 11 production fixtures passed: capacity 4 at positions 0,1,3,4,5 and
capacity 128 at 0,1,127,128,129,2048. Analytic consumed half-query operands
and full-cache bits matched exactly; independent chronological scores and
outputs passed fixed abs 2e-5 + rel 2e-4 bounds. All eight actual compiled
count=1/wrong-store-slot mutant fixtures were rejected, after the production
baseline passed. CPU reference controls also passed. Exit 0, no SKIP.

Full receipt and raw log:
`~/tmp/ds4-min/20261007-raw-attention-capacity-audit/METAL-PHYSICAL-RECEIPT.md`;
log SHA-256 `715d64b1babda568e00b6f318dbeb96ab87fe3e85b1768d88adb147f00ee9449`.
This qualifies the bounded raw pipeline and ring wrap only, not full-model,
compressed attention, grouped prefill, performance, or CUDA parity.

## Bounded Metal selected-Q4: qualified

Signed `275071b82c646cc4d2e88bc057eb7d2168b6e13d` passed one model-free
gpu-lock block on the same M4 Max/macOS device above. Binary SHA-256:
`faab8edd7d066234caa75754937e49c7d9c09cc0afc4cdab8df8729d0fd670dd`.
Common and slotted production routes each passed 42 independent float64
oracle values at fixed abs 1e-4 + rel 1e-4. Both actual compiled
high-scale-nibble mutant routes violated that same oracle on 42/42 values.
Four shape negatives and the restored-production invalid-ID write negative
passed. Five named subtests, no SKIP/FAIL, exit 0.

Receipt: `~/tmp/ds4-min/20261007-raw-attention-capacity-audit/Q4-PHYSICAL-RECEIPT.md`.
Raw-log SHA-256:
`3a2e668edef60c8efdaf82a274eac207f858b9046cfa0721ba959dff2d6fd622`.
This does not qualify a full-model Q4 splice, compressed attention, ownership
transactions, performance, or full parity.

## Bounded CUDA raw-attention checks

Select the actual test names, retain verbose non-SKIP output, and run under
gpu-lock:

```
-test.v -test.count=1 -test.run=^TestCUDA(QualificationRawAttention|RawAttentionFastBits|AttentionScoreParallelBits)$
```

On 2026-10-07, a clean `354ebe99979915eb8a971c794163ca6c89bd6d8e`
binary on GB10 sm_121 passed the qualification oracle at capacity 4 positions
0..10 and capacity 128 positions 127..129 (plus eight invalid-shape cases per
capacity). The bounds remained abs=2e-5 and rel=2e-4. The fused/unfused exact
cache, score and output comparison passed all seven positions
0,1,63,127,128,129,2048.

The separately selected parallel-score test failed all 15 exact normalized
score comparisons: positions 127..129 crossed with compressed counts
1,512,513,0,513 masked. Its compiled dot-index mutant was rejected as required.
The overall block returned rc=1; no numerical retry or bound change followed.
This is not an overall PASS or a full-model parity receipt.

The reference/mutant compile helper uses virtual `WithArch` without fast math;
production targets physical sm_121 with `--use_fast_math` and extra device
vectorization. This compile-contract mismatch is a source-backed, untested
fixture hypothesis, not an established explanation for the failures.

Full receipt and verbose log are retained at
`~/tmp/ds4-min/20261007-raw-attention-capacity-audit/` and on DGX at
`~/tmp/ds4-lib-workflow/20261007-raw-capacity/`. Log SHA-256:
`774de441cc1041f1f88df736662e0f799c419684faa36fe548a93fc552640a3c`.
Binary SHA-256, checked before execution and after failure:
`fb3576275baa549fdc712af144f06d8bf00d4299c191c71b68f0cc0090a0c503`.

## Bounded native Metal State compensation: qualified

On 2026-10-07, one gpu-lock block on Apple M4 Max qualified native Load
compensation using tiny nonempty shared MTLBuffers and a synthetic model
identity. The production decoder changed both HC fields before an injected
error. Successful compensation restored the independently saved native bytes
and cache metadata; failed compensation joined the primary and rollback
errors with ErrDesynchronized and suppressed logits. Both baseline branches
passed. The skipped-rollback mutant failed at the intended assertion,
`native shared-buffer rollback changed saved state`.

The tested source was clean base373e43acb3d30e1812ffbe6c89837223c9a8f523
plus one test file (SHA256
1bb20efc0a30cda803196d19d51d66d14fae2002cd73e164bb7998f7da79ed6d).
That test later landed signed as72027036da01226d838134c9d556d32b289c5396;
the later commit is not separately requalified. Baseline binary SHA256:
58f45775849311ade09475ed8a762fb405933352fbba837ffb82f2c24768b107.
Mutant binary SHA256:
36539655daa895f536e6e048cca0687e842bb00593239d903b84e20ad407bf72.
Receipt and raw logs:
`~/tmp/ds4-min/20261007-metal-state-qualification/SMALL-TERMINAL.md`.

This is a native codec/shared-buffer copy receipt, not GPU queued-work barrier
qualification, real-model continuation, request independence, or Metal parity.
The full Q2 IndependentStateGate remains required. Its latest single locked
read-only admission sample fell3.09GiB below the unchanged model+16GiB rule;
no model execution or reserve relaxation followed.
