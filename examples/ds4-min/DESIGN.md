# ds4-min contract

`ds4-min` is a single-model command for the pinned DeepSeek-V4-Flash Q2
checkpoint. It is not a server, a generic GGUF runner, or a compatibility layer
for the broader ds4 feature set.

## Command

```
usage: ds4-min -model file [-backend auto|metal|cuda|cpu] [-prompt text | -prompt-file file] [-tokens n] [-ctx n]

Load one DeepSeek-V4-Flash Q2 model and generate a non-thinking response.

  -backend string
        execution backend: auto, metal, cuda, or cpu (default "auto")
  -ctx int
        session context length (default 4096)
  -model string
        path to the pinned GGUF model (required)
  -prompt string
        user prompt; conflicts with -prompt-file
  -prompt-file string
        read the user prompt from file; conflicts with -prompt
  -tokens int
        maximum generated tokens (default 16)
```

If neither prompt flag is present, the command reads the prompt from standard
input. Extra arguments, unknown backends, conflicting prompt flags, unreadable
input, and non-positive token or context limits are errors. `auto` selects Metal
on darwin/arm64 or a usable CUDA device on Linux. It never falls back to CPU;
CPU must be selected explicitly and is a correctness reference, not a
performance target.

## Semantic boundary

One loaded model is immutable. One session is a mutable token timeline.
Synchronizing a full prompt evaluates only its suffix when the current token
sequence is a prefix. A divergent prompt rewinds to the common prefix and
rebuilds subsequent state. After synchronization, logits predict the next
token. Generation takes their deterministic argmax, checks the stop-token set,
emits the token bytes, and only then evaluates a non-stop token. It never
evaluates past EOS.

The stop-token set is derived from the model's tokenizer metadata, never from a
hard-coded id. It contains the vocabulary's `<｜end▁of▁sentence｜>` token
together with each of `tokenizer.ggml.eos_token_id`,
`tokenizer.ggml.eot_token_id`, and `tokenizer.ggml.eom_token_id` that the file
carries. Every member must be typed as a control token in
`tokenizer.ggml.token_type`; anything else is an error. For the pinned
checkpoint this derives exactly one id, 1, so the set matches the oracle's
single-EOS stop rule.

The only supported renderer is non-thinking chat:

```
<｜begin▁of▁sentence｜>[system]<｜User｜>[user]<｜Assistant｜></think>
```

The tokenizer and renderer must match the released
`encoding/encoding_dsv4.py`; whitespace substitution and generic Jinja
rendering are unsupported. That file is not present in the ds4 repository at
`b0a147a`. Before Slice 3 starts, capture the exact released source bytes and
SHA-256 and freeze its BPE vocabulary, merge ranks, special-token IDs, and
pre-tokenization regular expressions. The model file's tokenizer metadata must
agree with that receipt.

The accepted architecture has 43 layers and a 128-token raw attention window.
Layers 0 and 1 use raw attention. Even layers from 2 use ratio-4 compressed KV,
indexer KV, and at most 512 selected compressed rows. Odd layers from 3 use
ratio-128 compressed KV. Router bias may select experts, but mixture weights use
the selected unbiased logits.

## Pinned identities

The source oracle is git commit
`b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd` in `github.com/antirez/ds4`, read
only through `git show` or a clean archive. The conflicted ds4 checkout is never
an input.

The intended model is Hugging Face repository
`antirez/deepseek-v4-gguf`, file
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
No marketing alias identifies a model, and a different checkpoint is not a
substitute.

This section previously said the file was not available on the local Apple
host, so byte length, SHA-256, GGUF metadata digest, quantization manifest, and
tokenizer digest remained deliberately unset, and that model execution and GPU
implementation were blocked until they were captured. That is no longer true.
The file was obtained and the identities are captured in `gguf.go`:

    byte length                86720111488
    SHA-256                    ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
    GGUF metadata manifest     857d48d201da0faaaa6f2af0cdf0aa5fef523323217e29a902579130c24a32b5
    tensor manifest            0bcce58e7234ba89f247bfab49387121f375e8322d843fc166fb95ed12db99ca
    quantization manifest      b07c8ece42d1ff0273a8b7db85514e66ae6aca48dc55ea5b33d50f8f02380788
    tokenizer manifest         d58f8cbcd40bfa9319e0954e01ad619c179567cf001258423713d3d604998eec

The manifest digests cover, respectively, every metadata key, every tensor's
name/shape/type/offset/size, the per-quantization-type byte and tensor totals,
and the `tokenizer.`-prefixed metadata alone, each serialized under the schema
tag `ds4-min-gguf-manifest-v1`.

The enforcement is deliberately split. `validatePinnedProfile` checks the byte
length and the four manifest digests, and every load path calls it: the command
in `main.go`, the Metal runtime, and the CUDA runtime. Full-file SHA-256 is
*not* on that path. It lives in `validatePinnedModel`, which is called only
from `TestPinnedProfile`, because hashing 81 GiB on every run is not a load-time
cost the command accepts. So a load rejects any file whose header does not match
the pinned profile, while byte-for-byte identity of the tensor payload is
established by that opt-in test, not by the command.

The identity gate is therefore satisfied. Physical execution receipts are a
separate gate and are governed by the Evidence section below.

## Scope

The implementation stays in `examples/ds4-min`, with a private backend
interface. It does not use cgo, load or invoke ds4, silently fall back to CPU,
or claim arbitrary GGUF support. Server APIs, tools, vision, speculative decode,
batching, distributed execution, shared prefix caches, ROCm, training, and
million-token operation are outside this contract.

## Evidence

Every receipt binds the ds4 commit and binary digest, compiler, model and
metadata identities, prompt bytes and digest, flags, backend, physical device,
driver and OS, output hashes, timing availability, and process exit status.
Prompt token IDs, bounded logits/top-20, greedy token IDs, stop point, and
decoded bytes are compared. Apple Metal and NVIDIA CUDA are independent gates:
missing hardware or model is `UNMEASURED`, an executed mismatch is `FAIL`, and
one physical pass without the other is `PARTIAL`. Compilation, device
visibility, skipped tests, plausible output, and zero-sample profiles do not
establish physical execution.

## CPU fixture scope

The deterministic CPU fixture evaluates separate learned attention and FFN mHC
control projections with the pinned 20 Sinkhorn iterations,
attention and indexer compressor KV/gate/APE/norm projections, ratio-4 rolling
state, ratio-128 boundaries, index scoring, MoE routing, and output logits in
float32. It includes the attention sink and inverse rotary transform. The
pinned checkpoint's FP8 KV and indexer QAT round trips remain part of the
quantized GPU/full-model stage; the float32 fixture neither simulates them nor
claims quantized parity.

A second fixture stacks six tiny layers so the layer-type alternation is
covered: layers 0 and 1 are raw, layers 2 and 4 run the dual-lane fine
compressor with an indexer and a capped row selection, and layers 3 and 5 run
the single-lane coarse compressor with no indexer. Three constants are scaled
so a sixteen-token sequence can cross them: the raw window is 2 rather than
128, the coarse ratio is 8 rather than 128, and the selected-row cap is 1
rather than 512. The fine ratio 4, the alternation rule, and the indexer rule
are the real ones. The stack is written as a real GGUF file into the test's
temporary directory and reloaded through the GGUF parser, so tensor names,
metadata, and float32 tensor layout are exercised rather than bypassed. That
file is synthetic, so `validatePinnedProfile` rejects it; the test asserts the
rejection and loads through `readGGUF` directly. The gate on the command path
is unchanged. The stack fixture is float32 and unquantized: it does not cover
the pinned checkpoint's quantized tensor types, its tokenizer, or its 43-layer
and 128-token scale.

## Slice 0 review dispositions

- `KEEP`: the narrow command, immutable-model/mutable-session split, sync and
  generation order, pinned filename, clean oracle commit, compressed-attention
  outline, independent physical gates, and fail-closed missing identity.
- `REWORK`: the reviewer identified tokenizer pre-tokenization drift as a risk.
  The contract now requires the separately released encoder source and its
  digest before Slice 3; no regex is inferred from the absent model.
- `REFUSE`: execution or GPU approval while the checkpoint identity and runtime
  receipts are absent. Also refuse the reviewer's suggested split between stdin
  and chat rendering: stdin, `-prompt`, and `-prompt-file` are three transports
  for the same user-message bytes, so each feeds the same exact renderer. The
  identity gate is `BLOCKED`; Metal, CUDA, and full-model execution remain
  `UNMEASURED`.

NotebookLM's architecture response duplicated and garbled portions of its
quantization discussion. Those fragments are not evidence and were not used.
The filesystem checks above decide these dispositions.
