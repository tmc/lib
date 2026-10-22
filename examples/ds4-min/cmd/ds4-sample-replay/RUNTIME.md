# Seeded runtime admission

The retained September 30 run is not a sampled-stream control. Both arms
used the logprob observation path, which selects argmax without consuming
the RNG. Its numeric failure remains valid; its name does not prove seeded
generation coverage.

Before collecting another runtime comparison, require both runners to emit
the token selected by the live sampler and its RNG state immediately after
selection. Do not reconstruct this transcript from argmax or decoded text.
Capture raw logits immediately before that same selection, including a stop
token decision. Recording must not change the selection path.

Pin the following before acquiring gpu-lock:

- Clean C source revision 11e8cad, source and executable SHA-256; clean Go
  revision, executable SHA-256 and `go version -m` showing modified=false.
- Model SHA-256 ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0.
- The same 2048 int32 little-endian prompt IDs and their SHA-256, context
  4096, temperature 1, top-p 1, min-p 0.05, top-k 0, seed 12345.
- MTP off in both runners; C also sets DS4_MTP_SPEC_DISABLE=1. No logprob
  observation flag, since that flag replaces sampling with argmax.
- A fixed maximum of 16 decisions for the historical-shaped diagnostic, or
  64 for Spec 03 section 5.4. An early stop requires an explicit stop-token
  receipt, its decision index, and the captured final vector; a missing row
  is not evidence of a stop.
- Device, driver, toolkit, context and route flags, plus complete process
  exit status and raw stdout/stderr. Keep every failed attempt.

First run C/C with identical settings, then a predeclared arithmetic-order
control. The control must prove that its changed route executed. Measure its
per-step spread; do not invent a nonzero spread from a bit-identical control
or enlarge an existing numeric bound after observing Go. A bound that does
not exceed the measured control spread gives NO_VERDICT, not PASS.

Replay the exact captured vectors with `ds4-sample-replay -steps N corpus
grid`. This checks the host sampler on the vectors actually consumed by
each implementation, independently of the model. The corpus must contain
contiguous steps 0 through N-1 per case. Each case resets its RNG to the
specified seed; different cases never share an RNG stream.

Report separately: sampler equality on identical vectors; live token and
RNG agreement through the first different logit vector; numerical F5/F6
agreement; and F8 control coverage. Token disagreement after different
logits is not sampler failure, and sampler agreement is not numerical parity.

Runtime collection remains unqualified until both runners expose the live
sampled-token/RNG transcript described above and an observer-off/on control
shows unchanged generated tokens and final logits. This document is a
proposed physical protocol, not a hardware receipt.
