# Live head observer: proposed physical block

This is a diagnostic, not a performance mode or a numerical parity gate.
No device partial sums are produced. The packet comparison reports differences
without adopting a new tolerance. Source and host controls do not qualify GPU
execution.

## Build admission

Build Go from the observer commit in a clean isolated clone; retain `go version
-m`, `vcs.revision`, `vcs.modified=false`, executable SHA, source-file SHA
manifest, tracked module files and build argv. Do not use the dirty shared tree
as the physical binary. The source manifest file itself is hashed in the packet;
its relation to the executable still needs this external build receipt.

Copy the pinned 11e8cad oracle and its retained objects into a new scoped overlay
directory. Never patch the owning upstream directory. Check that
`head_overlay.patch` applies exactly, copy `head_observer.h` into `cuda/mmq/`,
then rebuild only `cuda/mmq/ds4_mmq.o` using the owning Makefile rule:

```
make -n cuda/mmq/ds4_mmq.o CUDA_ARCH=sm_121a
make cuda/mmq/ds4_mmq.o CUDA_ARCH=sm_121a
make -f Makefile -f observe.mk sample-observer CUDA_ARCH=sm_121a \
  ORACLE_DIR=. SAMPLE_OBSERVER_SOURCE=observer.c
```

Use the copied `observer.c`/`observe.mk`; inspect the dry run before building.
Retain source/object/compiler/header manifests and SHA256 of the linked binary,
and show the replacement object is in the actual link. The build recipe in
observe.mk, not a guessed linker command, owns the executable. The existing
oracle object/compiler receipt must agree with the chosen architecture/toolkit;
if it does not, HOLD instead of mixing incompatible objects.

## Exact common fixture

Model: retained `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
Prompt: the retained 2048 IDs, int32-LE
`5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380`.
Context 4096; 16 actual decisions; seed 12345; temperature 1, top-p 1,
min-p .05, top-k 0; ThinkNone; MTP disabled. Keep the admitted arithmetic
routes unchanged and label their differences diagnostic. No timing inference.

Set `SCOPE` to a fresh persistent project-scoped directory. Logs, source
manifests, existing C hook files and sampled decision directories are outside
each packet directory. Packet directories contain only the manifest and its
sealed stage/route files; extra files fail admission.

Per C observer-on arm:

```
DS4_HEAD_OBSERVER_PREFIX=$SCOPE/C1/packet/head
DS4_HEAD_OBSERVER_MODEL_SHA256=ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
DS4_HEAD_OBSERVER_SOURCE_MANIFEST=$SCOPE/c-source.sha256
DS4_METAL_GRAPH_DUMP_PREFIX=$SCOPE/C1/hooks/c
DS4_METAL_GRAPH_DUMP_NAME='result_hc result_norm result_output'
DS4_METAL_GRAPH_DUMP_LAYER=43
DS4_METAL_GRAPH_DUMP_POS=0
sample-observer MODEL PROMPT.i32 4096 16 12345 $SCOPE/C1/decisions
```

The observer requires its copied live normalized input to have exactly the
same hash as the existing C result_norm hook, then seals HC and final head
logits immediately after session_sync,
before any generated token is evaluated. The live Q8_1 overlay copies the
selected folded/scratch activation before the consumer. A raw/fallback route
does not produce Q8_1 records and is unsupported, not reconstructed. The C
hook uses literal POS=0 for the head, independent of actual prompt position.

Per Go observer-on arm:

```
DS4_HEAD_OBSERVER_PREFIX=$SCOPE/G1/packet/head
DS4_HEAD_OBSERVER_POSITION=2047
DS4_HEAD_OBSERVER_MODEL_SHA256=ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
DS4_HEAD_OBSERVER_SOURCE_MANIFEST=$SCOPE/go-source.sha256
ds4-min -backend cuda -model MODEL -ctx 4096 -prompt-tokens PROMPT.json \
  -tokens 16 -temp 1 -top-p 1 -min-p .05 -top-k 0 -seed 12345 -nothink \
  -dump-logits $SCOPE/G1/logits \
  -dump-sampled-trace-dir $SCOPE/G1/decisions
```

These literal spellings were checked against current parseConfig and the
retained admitted 0ffc runtime argv. Recheck the clean binary before execution.
Do not substitute argmax `-dump-logprobs` or speculative mode. Set
DS4_MTP_SPEC_DISABLE=1 identically in all arms. All observer-off arms unset
all HEAD_OBSERVER and existing C dump vars,
but retain sampled-decision observation so RNG/selection can be compared.

## Block order and stop rules

Only after coordinator review: one gpu-lock lease, exact order C0,C1,C2,G0,G1,G2.
0 = head observer off; 1/2 = head observer on with fresh paths (A/A).
Every arm uses the same newly built per-implementation binary. Rehash binaries
and prompt before/after. Retain exact env/argv, admission, rc and raw stdout.
C observer writes terminal raw logits to `DECISIONS-DIR.final.f32`, outside
the exact-count decision directory. All C arms pass a fresh decision directory
even when head observation is off. All Go arms retain -dump-logits plus the
actual sampled trace, with fresh paths.

Require 16 decisions with explicit limit/stop disposition; compare full raw
prefill/final logits, selected IDs, histories and before/after RNG within each
implementation. C0/C1/C2 and G0/G1/G2 must be byte-identical in those outputs.
Require C1/C2 and G1/G2 live stage bytes identical (path metadata may differ).
Failed admission, nonzero rc, missing route, missing stage, failed seal,
nonfinite vector or any off/on/A/A difference is HOLD with no retry.

Run host validation under gpu-lock:

```
ds4-sample-replay -head-pair $SCOPE/C1/packet/head.json $SCOPE/G1/packet/head.json
ds4-sample-replay -verify-decisions $SCOPE/C1/decisions -prompt-i32 PROMPT.i32 -steps 16 -seed 12345
ds4-sample-replay -verify-decisions $SCOPE/G1/decisions -prompt-i32 PROMPT.i32 -steps 16 -seed 12345
```

The model checksum is explicitly an external receipt assertion: no 86 GB
rehash is silently implied. Executable and source-manifest files are actually
hashed by the producers. The physical admission must bind the asserted model
receipt to the path used by each arm.

Go history is the actual accepted evaluation timeline. A pending group is
visible during its head evaluation and committed only after success. Failed
evaluation invalidates it. An admitted Reset or successful Load/Restore
invalidates it; rejected validation and a failed cache transaction followed
by successful rollback preserve it. Failed rollback invalidates and poisons
state. Invalidation occurs after cache replacement but before epoch advancement,
even if epoch advancement subsequently fails. Ordinary successful Eval-like
epoch transitions preserve accumulated history. Observer continuation at a
nonzero position after invalidation refuses missing history. A fresh Reset
plus replay is required. Test missing/duplicate/reordered history, partial
failure, reset, rejected transitions, successful rollback, failed rollback,
epoch exhaustion and restored-cache-without-history before claiming this contract.

## Interpretation

First compare HC, then normalized input, then Q8_1 scale/codes, then logits.
The block sum is retained but unused by the head consumer. If normalized input
differs, quantify that upstream mismatch first; quantizer differences on
different inputs do not localize the quantizer. A same-frozen-normalized-input
quantizer replay is a separate reviewed intervention. Host integer block-dot
or float64 scaled partials must be labeled reconstructed, not device partials.

Existing actual-sampler step0 logits differ and step6 tokens diverge; this
experiment does not erase those FAILs. It supplies the first measured head
boundary. Stop-token identity remains UNREAD unless a tokenizer stop receipt
is independently validated.
