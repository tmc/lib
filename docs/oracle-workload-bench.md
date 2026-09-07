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

## Qualified oracle references

Two single-repetition references have completed with passing before/after load
checks. These are oracle measurements, not port results or evidence of parity.

| Backend and host | Prefill tokens/s | Generation tokens/s | Load before / after | Limit |
| --- | ---: | ---: | ---: | ---: |
| CUDA, GB10 spark-31db | 661.18 | 17.73 | 1.00 / 1.77 | 5.00 |
| Metal, M4 Max m4max.local | 310.54 | 28.59 | 3.53 / 2.87 | 4.00 |

Both use 2048 prefill tokens, 128 generated tokens, context allocation 65665,
and prefill capacity 4096. Both use b0a147a **plus the retained compatibility
patch**, not pristine b0a147a. Their post-patch `ds4.c` hash is
`2403b881be7adbdb50f5901aabe2a08f35e7b34ed09f5b57f1b608843894393b`;
the backend implementations and binaries differ. The model hash is
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
on both hosts.

The Metal benchmark has an additional input-capture patch and retains the
actual consumed prefix with the registered hash above. The CUDA input binding
comes from a separate token dump and source-path review, not a prefix capture
inside the timed process. Do not describe the complete sources or input-binding
methods as identical.

Receipts under `evidence/worker-CODEX6E6A/` are `oracle-bench-2048` for CUDA
and `oracle-metal-bench-2048-window2` for Metal. Their `RESULT-MANIFEST.sha256`
hashes are respectively
`e322d67f79abd05032aa7031293540707f5dfed81302c2237946e4b7a07ddda0` and
`a975e8140a786185a4efddb178ec09b4f7f5270c5bd0b6726dae0dc73d54dd19`.
The earlier Metal receipt, `oracle-metal-bench-2048`, is disqualified and is
not included in the table. The difference between its raw rates and the
qualified window does not isolate a cause. One qualified repetition per
backend establishes neither statistical significance nor a port/oracle ratio.

## Physical acceptance

Physical CUDA attempts have timed out without `result.json` or `COMPLETE`;
there is no completed port 2048-token rate from those attempts. Retained receipts
are `evidence/worker-BABFB352/oracle-workload-current-20260907` and
`evidence/worker-BABFB352/cuda-compressor-defer-workload-20260907`. Their terminal
stack samples establish where execution was observed, not a rate or a cause.
Device health during those attempts is not established by retained evidence.

Before running, bind source and executable identities, pass the registered warmed bit-identity gate for the
production change, reserve the serial model slot, and use qualified-run with the
registered host limit before and after timing. Do not rerun the old unchanged
2048-token CUDA implementation, whose 55-minute timeout is already retained.
Use a fresh process per repetition and a bounded timeout. Retain go test -json
events; check positive reach of TestOracleWorkload with scripts/check-go-test.py
and the actual exit. A plain package PASS can mean this test skipped.

Finish builds, tests and source hashing that are external to the harness before
taking a qualified timing window. The slot owner must also avoid unrelated
CPU, disk and GPU work on that host until after qualification is recorded.
A slot reservation coordinates agents; it does not enforce host quietness.
Do not widen the registered load limit to admit a contended run.

On CUDA, retain `devicequalify` observations before and after the run as well
as the load qualification. The device probe checks driver initialization and
metadata responses, not kernel execution or continuous health during the run.
PCIe expectations are optional and require a relevant retained baseline; a
link reading alone is not a demonstrated transport fault on this GB10.

identity.json also binds the discrepancy receipt path/hash and the EOS vocabulary-token lookup
in the bound model. It binds host/backend, model, executable, prompt, corpus, context and
lifecycle. result.json retains generated IDs, timing windows and full prefill/
final float32 hashes. COMPLETE is written only after runtime Close succeeds and
results are written. Require COMPLETE, 2048 prefill tokens, 128 generated IDs,
positive finite durations, valid identities and positive test reach before
interpreting rates. Source manifests and qualification remain external evidence.
One observation cannot establish statistical parity.

Model-free controls cover exact prefix/calls/positions, EOS exclusion, ties,
timing windows, invalid inputs, non-finite and wrong-length logits, and runtime
errors. EOS-inclusive selection and 127-decode mutations must turn them red.
