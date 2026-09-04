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
GOWORK=off go test -count=1 ./ds4-min
GOWORK=off go test -race -count=1 ./ds4-min
GOWORK=off go vet ./ds4-min
GOWORK=off go test -run TestFixtureRegeneration -count=2 ./ds4-min
```

When the corresponding dependency surface changes, run its module-local tests
with `GOWORK=off` as listed in the approved workflow.

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

Use `PASS`, `PARTIAL`, `UNMEASURED`, `FAIL`, or `BLOCKED` exactly. At contract
freeze, the identity prerequisite is `BLOCKED` because the pinned GGUF was not
present locally and its exact identity could not be captured. Full-model,
Metal, and CUDA execution are therefore `UNMEASURED`.
