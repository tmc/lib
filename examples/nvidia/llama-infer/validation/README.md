# Transformers logit parity

TestHFParity compares llama-infer's full-vocabulary logits with Hugging Face
Transformers on a real checkpoint, at every prefill-chunk boundary and every
decode step of a fixed set of chat turns. Reference suites come from
validation/hf_reference.py.

Run reference generation and each engine/chunk in separate GPU-lock holds.
The script requires a fresh ancestor lock with a declared limit of
1,800 seconds; do not chain phases inside one hold. A phase has a
1,500-second payload budget, including its build, and GNU timeout kills its process group after a 15-second grace.
An engine's Go test also has a 24-minute timeout. A timeout is a failed
phase; its logs remain and the script does not retry it.

The following fragments illustrate the staged commands; GPU execution has
not been verified in this source lane. Set CHECKPOINT to the local HF
snapshot. Run them on dgx from the llama-infer package after reviewing the
checkpoint and source receipts:

```sh
export PYTHON=/home/tmc/ml-explore/mm-ref-venv/bin/python
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64
out="$HOME/tmp/lib-wt-llama-megakernel/$(date +%Y%m%d)-hf-parity"
GPU_LOCK_LABEL=hf-reference GPU_LOCK_EXPECT=1500 GPU_LOCK_LIMIT=1800 \
    /home/tmc/ml-explore/bin/gpu-lock -- \
    validation/hf_parity.sh "$CHECKPOINT" "$out" reference
```

Then run one engine/chunk per hold. For example:

```sh
LLAMA_HF_PARITY_GRAPH=1 \
GPU_LOCK_LABEL=hf-egraph-4 GPU_LOCK_EXPECT=1500 GPU_LOCK_LIMIT=1800 \
    /home/tmc/ml-explore/bin/gpu-lock -- \
    validation/hf_parity.sh "$CHECKPOINT" "$out" egraph 4
```

Repeat egraph for chunks `0`, `1`, `4`, `8`, `40`, `all`, with graph settings
`0` and `1`, to cover both eager and graph decode. `bf16` and `f32` use
chunk `0` and eager decode. This script is a bounded fidelity payload;
the coordinator's job driver still records before/after GPU admission and
PID ownership for qualification.

The reference phase writes `OUT/ref`. Engine phases require a completed
reference PASS marker, exit-code 0, and a suite checkpoint path canonically
identical to the requested checkpoint before building or launching. An engine phase creates
`OUT/ENGINE-graphGRAPH-chunkCHUNK`, which holds `env.txt`, the lock-owner
receipt, build/run logs, binary hash, exit code, and `report.json`. A
completed phase gets `PASS` only after validating that the native test was
not skipped and every reference case has positive, passing row coverage.
The marker describes that phase's selected tolerance, not stricter parity
or performance. An existing phase directory is not overwritten. Failed
reference directories are preserved; use a new task output for a new
reference attempt. The throwaway Go test executable is removed after the
run; its hash remains.

An exported source tree has no Git metadata: set `HF_PARITY_SOURCE` to the
coordinator's source-hash receipt. The phase records that receipt's hash;
the coordinator must separately verify that it describes the built tree.

Reference logit storage grows with rows times vocabulary size, at four
bytes per float32 value. Keep these tensor files out of the repository.

## Reference

hf_reference.py runs Transformers in float32 with eager attention and TF32
disabled. For each turn it records the chat-template token IDs and the
greedy continuation (up to 24 tokens, stopping at EOS), checks that the
cached generation matches a full causal pass, and saves that pass's logits
for every position from the turn's prefill start to its last generated
token. A follow-up turn's prefill start is the prefix it shares with the
previous turn's prompt and continuation: what a chat runner keeps in its KV
cache. Use REF_ARGS="--dtype bfloat16" only when the float32 model does not
fit; the suite records the dtype, and a bfloat16 oracle is weaker. Such a
suite has no float32-to-BF16 calibration, so BF16 engines require an
explicit prospectively chosen LLAMA_HF_PARITY_ATOL override; the script
does not derive a passing tolerance for it.

The fixed conversations cover arithmetic, a remembered name, translation,
follow-up turns, replay after reset, and a passage summary. The passage is
intended to exercise the tile GEMM path as well as narrow prefill; actual
row coverage is recorded in the report.

The suite records each file digest's source in `file_digest_sources`.
`hf_cache_blob_name` is an HF cache identity taken from a hexadecimal
blob name, without freshly hashing the weight contents; `sha256_content`
is a fresh content hash. Missing source labels in an older suite must not
be presented as proof of freshly verified weight contents.

The reference also runs the same sequences in bfloat16 and records, per
row, Transformers' own bfloat16 error against float32.

## Comparison

The test continues the KV cache from turn to turn, prefills each turn's new
tokens in chunks, then decodes the reference's generated tokens (teacher
forcing), including a final EOS. It compares the logits row after every
chunk and every decode step with the reference row at that position. Every
element must satisfy |ours − ref| ≤ atol + rtol·|ref|. Reported per turn:

- max_abs: the largest element error, and its position.
- max_row_rel: the largest row error divided by the row's largest |ref|.
  Equal all-zero rows have relative error 0; nonzero error against an
  all-zero row uses finite MaxFloat64 as an unbounded-relative-error
  sentinel. The numeric tolerance gate is evaluated separately.
- max_tol_ratio: the largest error divided by its allowed tolerance; ≤ 1 passes.
- max_bf16_ratio: the largest ratio of our row error to Transformers'
  bfloat16 row error at the same position, over rows with positive
  calibration error.
- top1_agree: rows whose argmax matches the reference's.
- greedy_agree / greedy_rows: of the rows that predict a generated token, those
  whose argmax is that token. All of them agreeing is what greedy decoding
  equal to the reference's requires.
- disagreements: each row whose argmax differs, with the reference's margin
  between its top two logits and our margin of our argmax over the
  reference's.

Tolerances:

- f32: the fixed float32 gate, atol 5e-3 and rtol 5e-4.
- bf16 and egraph: atol = the suite's measured maximum of Transformers'
  bfloat16 error against float32 (hf_bf16.max_abs in suite.json), rtol 0.
  These engines store BF16 weights; the egraph path also uses BF16
  intermediates and a generated BF16 KV cache. Passing means that no
  compared logit exceeds this suite-wide calibration bound.
  max_bf16_ratio separately compares each checked row with its positive
  calibration error. LLAMA_HF_PARITY_ATOL and _RTOL override either; values
  must be finite
  and nonnegative. Zero for both requests exact element equality. The BF16
  calibration bound is a different, generally weaker gate than the fixed
  float32 tolerance; it is not the same atol 5e-3 / rtol 5e-4 claim.

Chunks for the egraph engine (LLAMA_HF_PARITY_CHUNKS, default
0,1,4,8,40,all): 0 prefills token by token through the decode step
(-egraph-prefill=false); all prefills each turn's new tokens in one chunk
(the default); a row count prefills in chunks of that many rows. The CLI
rounds -egraph-prefill-rows up to a multiple of 64, so 1, 4, 8 and 40 are
harness chunkings; they run the kernels a short last chunk runs.

## What a pass does not establish

- llama-infer's tokenizer and chat template are not tested: the test feeds
  the reference's token IDs.
- Sampling other than greedy, and free-running generation: decode is teacher
  forced. Greedy agreement on every row implies free-running greedy output
  equals the reference's; a disagreement means it would diverge there.
- Prompts, chunkings and conversations other than these, or contexts
  longer than the longest turn.
- Other engines' flags (-egraph-mega, -fuse-decode, speculative decoding).
