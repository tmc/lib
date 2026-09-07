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
