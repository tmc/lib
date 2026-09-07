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
corpus of at least 2048 IDs), and DS4_ORACLE_WORKLOAD_DIR (nonexistent directory).
The port tokenizer must reproduce the retained first 2048 IDs from the raw text.

The corpus must also be independently verified against the oracle tokenizer for
this prompt and model. The port-side check cannot establish that provenance.
The retained CUDA oracle reference has no token-ID dump yet; corpus provenance
must close before a physical port result is called workload-matched to it.

## Physical acceptance

No physical run has been made with this harness. Before running, bind source and
executable identities, pass the registered warmed bit-identity gate for the
production change, reserve the serial model slot, and use qualified-run with the
registered host limit before and after timing. Do not rerun the old unchanged
2048-token CUDA implementation, whose 55-minute timeout is already retained.
Use a fresh process per repetition and a bounded timeout. Retain go test -json
events; check positive reach of TestOracleWorkload with scripts/check-go-test.py
and the actual exit. A plain package PASS can mean this test skipped.

identity.json binds host/backend, model, executable, prompt, corpus, context and
lifecycle. result.json retains generated IDs, timing windows and full prefill/
final float32 hashes. COMPLETE is written only after runtime Close succeeds and
results are written. Require COMPLETE, 2048 prefill tokens, 128 generated IDs,
positive finite durations, valid identities and positive test reach before
interpreting rates. Source manifests and qualification remain external evidence.
One observation cannot establish statistical parity.

Model-free controls cover exact prefix/calls/positions, EOS exclusion, ties,
timing windows, invalid inputs, non-finite and wrong-length logits, and runtime
errors. EOS-inclusive selection and 127-decode mutations must turn them red.
