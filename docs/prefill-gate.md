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
