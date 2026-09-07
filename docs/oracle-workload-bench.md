# Oracle workload measurement

`TestOracleWorkload` calls production `EvalBatch` and `Eval` directly. It measures
a fresh runtime's first 2048-token prefill at context 65665, followed by exactly
128 greedy non-EOS decode steps. It does not add a chat template, BOS, warm pass,
text rendering or text output. The oracle's prefill capacity 4096 fits this
frontier; the port chooses its internal group width.

The normal port CLI is a different workload: it wraps the prompt, stops on its
stop-token set and times decoding/output. This harness excludes only the model's
EOS ID and resolves ties to the lowest ID. Its selector checks finite values,
which adds host-side validation inside the overall generation window. First and
steady token times exclude selection, as in ds4-bench. Prefill/final full-logit
hashing is outside the measured windows.

## Required inputs

Set DS4_ORACLE_WORKLOAD_GATE=1, DS4_MODEL, DS4_ORACLE_WORKLOAD_BACKEND (cuda or
metal), DS4_ORACLE_WORKLOAD_PROMPT (raw text), DS4_ORACLE_WORKLOAD_TOKENS (JSON
corpus of at least 2048 IDs), and DS4_ORACLE_WORKLOAD_DIR (nonexistent directory), and
DS4_ORACLE_WORKLOAD_TOKENIZER_RECEIPT (retained discrepancy receipt).
The retained first 2048 IDs must match the registered oracle token hash below.

The port is evaluated on the oracle's tokenization, which its own tokenizer
would not produce. This is a numerical workload measurement, not a claim that
the two front ends agree. The first tokenizer discrepancy is at token 904:
the pinned C tokenizer treats non-ASCII punctuation as letter-like.

The registered 2048-ID prefix SHA-256 is
`5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380`,
over exactly 8192 bytes of little-endian uint32 IDs. It comes from the pristine
b0a147a CPU token dump of promessi_sposi.txt. All 21 recognized special markers
and NUL are absent, so that dump follows the benchmark's raw BPE path. The
harness rejects a different prefix before runtime creation and records the
binary prefix, encoding, count and hash separately from the original JSON
file hash. Both reference and port receipts must name this token hash.

## Physical acceptance

No physical run has been made with this harness. Before running, bind source and
executable identities, pass the registered warmed bit-identity gate for the
production change, reserve the serial model slot, and use qualified-run with the
registered host limit before and after timing. Do not rerun the old unchanged
2048-token CUDA implementation, whose 55-minute timeout is already retained.
Use a fresh process per repetition and a bounded timeout. Retain go test -json
events; check positive reach of TestOracleWorkload with scripts/check-go-test.py
and the actual exit. A plain package PASS can mean this test skipped.

identity.json also binds the discrepancy receipt path/hash and the EOS metadata
key in the bound model. It binds host/backend, model, executable, prompt, corpus, context and
lifecycle. result.json retains generated IDs, timing windows and full prefill/
final float32 hashes. COMPLETE is written only after runtime Close succeeds and
results are written. Require COMPLETE, 2048 prefill tokens, 128 generated IDs,
positive finite durations, valid identities and positive test reach before
interpreting rates. Source manifests and qualification remain external evidence.
One observation cannot establish statistical parity.

Model-free controls cover exact prefix/calls/positions, EOS exclusion, ties,
timing windows, invalid inputs, non-finite and wrong-length logits, and runtime
errors. EOS-inclusive selection and 127-decode mutations must turn them red.
