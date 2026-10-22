# Isolated C sampler observer

This runner uses public `ds4_session_sample` from pinned oracle 11e8cad.
Source inspection confirms it invokes `sample_top_p_min_p` on live session
logits with the supplied RNG. The runner supplies temperature 1, top-k 0,
top-p 1 and min-p 0.05. It sets DS4_MTP_SPEC_DISABLE=1 and never invokes a
speculative API or the logprob observation path.

Input is raw little-endian int32 token IDs, not text or permissively parsed
JSON. Convert the pinned prompt JSON once, hash both files, and verify that
decoding the binary IDs reproduces the JSON exactly before runtime admission.
The executable accepts model, prompt.i32, context, decisions (16 or 64),
positive seed, and a fresh output directory. A final argument of `-` disables
the observer while preserving sampling. Both modes print the actual selected
tokens, RNG states and explicit terminal disposition. Stop tokens are recorded
but not evaluated or appended to model history, matching Go's loop.

Build in a new scoped output directory using the oracle's owning Makefile,
without editing that source checkout. Supply absolute paths:

```
make -f "$ORACLE_DIR/Makefile" -f "$OBSERVE_MK" sample-observer \
  ORACLE_DIR="$ORACLE_DIR" SAMPLE_OBSERVER_SOURCE="$OBSERVER_SOURCE" \
  CUDA_ARCH=sm_121
```

The core object paths and any Makefile relative prerequisites must resolve to
the retained owning build. Inspect the dry-run first; if it tries to rebuild
in the oracle checkout or references missing relative sources, stop and use
the pinned build's exact object/link receipt instead. Do not infer a working
link recipe from syntax validation.

Before model comparison, run C observer-off/on under one gpu-lock lease.
Require every printed decision, RNG state, stop disposition and final logits
SHA-256 to match. Then require Go observer-off/on to match likewise. Raw output
is deliberately diagnostic until these controls pass. Compare matching-schema
step JSON and .f32 records only after verifying model/prompt/binary/source
hashes and the explicit expected terminal count or matching stop decision.

Local compile-only validation used the pinned ds4.h and ds4_ssd.h with
`cc -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror
-Wno-unused-function -fsyntax-only`. No CUDA link or runtime qualification
is claimed. The SHA-256 implementation is copied verbatim from the campaign's
retained matched-c-range-runner.c; record that source hash in the build receipt.
