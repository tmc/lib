# Opt-in Metal grouped verifier

`DS4_METAL_SPECULATIVE_GROUPED=1` selects a real grouped command schedule for
1–8 teacher-forced tokens. It admits context at most 128 and refuses collectors,
steering, invalid tokens/extents and host-indexer fallback. Default is scalar.

The device encodes embeddings, all 43 layer operations, and each residual row
in one ordered command buffer. Scratch is reused only after an ordered blit
preserves the row. Output normalization and the F16 HC projection include all
rows. After one wait, the existing float64 host sigmoid and f32 accumulation
order are retained. One second command buffer computes all-row normalization
and Q8_0 logits. This is grouped device scheduling and batched readout, not
multi-row layer FFN fusion. It never calls scalar Eval to produce grouped rows.

An execution owner tracks both command buffers and all five scratch buffers.
Unsubmitted encoders are ended before the HC claim is cleared. Submitted work
must reach a terminal device status before any scratch is released. Completion
errors are joined; nonterminal work retains scratch and poisons the runtime,
whose Close refuses to release the arena until the work is terminal. The buffer
Free API has no error return. Actual post-encoding faults at both buffers and a
post-completion error have bounded model-free controls; full-model frontier
rollback under both faults belongs to the physical gate.
The retained-owner/Close-retry test uses a synthetic status classifier after an
actual device buffer completes. It checks ownership and Restore refusal, not
real nonterminal device recovery.

Verification reads the pre-draft frontier for draft 0 and output row i-1 for
draft i. It still evaluates the last draft, preserving final cache state.
CommitPrefix restores the original frontier and replays exactly the accepted
prefix with scalar Eval, including zero/full acceptance. Errors restore the
frontier; no unsupported group silently falls back to scalar work.

`DS4_METAL_SPECULATIVE_GROUPED_GATE=1` enables the physical registered-model
self-oracle. For n=2 and n=8 it requires every logit bit, terminal serialized
state and two grouped submissions to match the independent scalar route.
All 12 accepted prefixes and their continuations must match serialized state;
78 proposal-frontier top-1 values are checked. Finite/shape checks fail closed.

Full-model hardware qualification and speculative performance are UNREAD.
The gate requires the exact pinned model, live gpu-lock and model+16 GiB memory
admission. Bounded model-free embedding/copy and CPU row-collapse tests do not
establish transformer equivalence or a performance improvement.
