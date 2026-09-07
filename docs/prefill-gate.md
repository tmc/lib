# Exact batch gate

`TestPrefillCapture` is a capture producer for comparing old and new `EvalBatch`
implementations on the same backend. It does not claim oracle equivalence or
measure throughput. The CUDA production gate `TestCUDAProductionHash` checks token 0 at position 0
before and after Reset, at context 1024, against `e1500...`. It is opt-in through
`DS4_CUDA_PRODUCTION_HASH_GATE=1` and `DS4_MODEL`. Retain that check separately;
it does not exercise `EvalBatch`.

Freeze both source trees from one named baseline, with the candidate patch
recorded explicitly, then build the same `prefill_gate_test.go` into each. Compare
the complete recorded source set against the intended tree before launch; a
previous model-free test copy may contain older production kernels even when
the gate file is current.
The test compiles for Darwin arm64 and Linux arm64. Those builds do not establish
hardware correctness. The n=2 CUDA baseline/candidate capture and fixed
continuation passed exact comparison in the retained
`evidence/worker-BABFB352/prefill-n2-layer-major-20260907/` receipt. Reversing
within-layer token order made the comparator fail. Cases 64 and 2048 remain
unmeasured; the n=2 result does not cover them.

The input is a retained JSON array of at least 2048 token IDs in [0,129279]. Use
the same tokenizer output for both candidates; do not independently regenerate
it between arms. Choose and retain one fixed continuation token. The test hashes
the model and executable, uses context 2049, and emits 129280 finite logits in
little-endian float32 form at both prefill and continuation boundaries.

Run one model process at a time under the host owner's reservation. These are
identity captures, so they do not require a quiet-host timing qualification.
Throughput measurements require the authoritative qualified-run wrapper and its
pre-registered host limit. Start with the two-token case:

```
DS4_PREFILL_GATE=1
DS4_MODEL=/absolute/model.gguf
DS4_PREFILL_BACKEND=cuda
DS4_PREFILL_TOKENS=/absolute/tokens.json
DS4_PREFILL_CASES=2
DS4_PREFILL_CONTINUATION=0
DS4_PREFILL_DIR=/absolute/new-capture-directory
```

Export those variables for an explicitly selected binary invocation:

```
/path/to/reference.test -test.run '^TestPrefillCapture$' -test.count=1 -test.timeout=15m
```

Use a different output directory for each binary. Capture directories must not
exist; their parent directories must exist. Each case runs twice: a warm batch
and continuation, then Reset and a captured batch and continuation. Only one
runtime exists. `COMPLETE` is written after successful Close. A partial directory
is diagnostic evidence and must not be used as a comparison pass.

Compare the two-token captures:

```
python3 -B scripts/compare-prefill.py /path/to/reference /path/to/candidate --cases 2
```

This result is explicitly scoped to two tokens. Extend to 64 after the two-token
gate passes. The complete suite requires ordered cases `2,64,2048`, including
continuation vectors for each. The comparator rejects missing files, incomplete
runs, mismatched lifecycle/backend/model/input identity, nonfinite values,
changed bits including signed zero, and directory self-comparison.

Do not rerun the old 2048-token CUDA path unchanged: its retained run already
failed to complete in 55 minutes. The coordinator explicitly prohibited another
such run. A complete old/new 2048-token comparison therefore remains pending a
usable reference capture and scheduling decision; the new test does not supply
that missing evidence. Do not report the full gate passed because smaller cases
or compilation pass.

Model-free controls:

```
python3 -B scripts/compare-prefill_test.py
python3 -B scripts/qualified-run_test.py
```

Ten comparator controls and eleven qualification controls passed. A missing-model
opt-in invocation failed before loading, as required. Binary and source identities
and logs are in `evidence/worker-CODEX6E6A/iteration1/` outside this checkout.

## Ragged batch groups

The producer and comparator also accept explicitly selected cases `65` and
`66`. For width-eight implementations these exercise a singleton and a
pair after eight complete groups. Use `DS4_PREFILL_CASES=64,65,66` for both
capture arms and `--cases 64,65,66` for comparison. These cases retain the same
warm/reset lifecycle, fixed continuation, context 2049, and minimum 2048-ID
corpus. The default comparator suite remains `2,64,2048`; a ragged-only result
is scoped and does not establish the 2048-token gate.

Use distinct tokens at corpus indexes 63, 64 and 65. A useful production-path
mutation swaps indexes 63 and 64 while preserving their positions, after the
last complete group meets the tail. Both 65 and 66 contain that boundary.
For n=66, additionally reversing indexes 64 and 65 probes ordering within the
last pair. Apply mutations to candidate execution only, keeping the retained
input corpus and identities unchanged; changing the corpus itself tests the
identity check rather than runtime token ordering. Each mutated capture must
complete and then fail the exact logit comparison. No comparator can guarantee
that an arbitrary runtime mutation changes the observed logits; retain the
physical negative result rather than assuming it.

## Ratio-128 emission cases

Select `DS4_PREFILL_CASES=128,129` and compare with `--cases 128,129` to
exercise the first ratio-128 compressed row and subsequent consumption.
With zero-based positions, the 128-token prefill reaches emission at position
127; its fixed continuation evaluates position 128. The 129-token case also
exercises a singleton tail after sixteen full width-eight groups, followed by
continuation at position 129.

Use the same canonical token corpus and fixed continuation in both arms. These
cases retain context 2049, the warm/reset lifecycle, and exact finite float32
comparison for both prefill and continuation. They are scoped checks and do
not replace the default `2,64,2048` suite or establish 2048-token identity.
