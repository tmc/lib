# Trajectory advice consumption

This package consumes the existing corpus and sidecar schemas; it does not run
experiments or replace their native validators. It preserves static STOP,
component timing and full-entry outcomes separately. There is no score-ratio
heuristic for a lead, miss, mislead or causal claim.

JSON identities require valid UTF-8 and paired UTF-16 surrogate escapes. Invalid
Unicode is rejected before decoding; it is never silently replaced by U+FFFD.
Valid multibyte text and paired escapes retain the Python sidecar identity.

Import actual retained records without inventing advice attribution:

```
GOWORK=off go run ./x/cudaprof/advisecorpus/cmd/advisecorpus -root "$R" \
  "$R/methods/top20-20261028/corpus/revision-3/records.ndjson" \
  "$R/methods/top20-20261028/corpus/new-closed-trajectories-1/records.ndjson" \
  "$R/methods/top20-20261028/corpus/new-closed-trajectories-1/static-closures.ndjson"
```

All 27 currently lack an admitted sidecar and remain `NOT_EVALUATED`. A missing
finding cannot be retrospectively labeled a miss. Once an exact reviewed sidecar
exists, the same command accepts `-link /path/to/link.json`; it evaluates only
that record ID and fails without output if the ID is absent or any binding fails.
No existing native capture is claimed to implement the new validator DTO.

The API calls `advise.AnalyzeWithEvidence` on retained capture bytes through
`Analyze`. The selected source must occur in a named translation-unit source
role of that capture, not merely as an arbitrary attachment or matching symbol.
For a Python wrapper without a retained source-to-compilation mapping it returns
an error; it never equates raw Python, CUDA and image digests. The supplied
`cudaevidence.Source` remains a retained association, not hardware attestation.
`TestAnalyzeCallsRealAPIWithoutNativeClaim` exercises the real API with an
explicit compiler-only CPU fixture and no profile samples. No CUDA compiler,
GPU, profiler or native collector runs in these tests.

`Match` matches an explicitly enrolled finding, recommendation and kernel
exactly. It returns indices, not an outcome. A reviewer must retain the unchanged
mechanism and pre-decision chronology in the existing sidecar/decision receipt.
`Consume` checks those records and preserves suspected findings as suspected.
The labels `STATIC_*_LOGGED_NOT_RUNTIME_WIN` and
`RUNTIME_*_LOGGED_NOT_CAUSAL_PROOF` describe a scoped nonblind log, not independent
proof that an optimization was caused by advice.

Runtime attribution requires both per-arm proofs bound to the exact context,
case, boundary and source pair; complete/loss-free capture; postprofile numerical
PASS; retained validator source bytes; and a strict
`corpus-advise-validator-receipt/v1` PASS receipt agreeing on every binding. The
input capture must equal the selected arm's raw capture. Empty validator JSON,
malformed digests, unrelated captures and static-to-runtime promotion fail.
This is the repaired Python owner contract from
`setup-jit-contract-1/advise-revision-2`, not a guessed NCU/CUPTI API return type.
There is currently no native producer for that DTO; runtime attribution remains
HOLD until one is separately implemented and reviewed.

Run host controls with:

```
GOWORK=off go test ./x/cudaprof/advisecorpus/...
```

Tests use CPU mock artifacts, explicitly marked as such. Their positive runtime
branch checks DTO join mechanics only; it is not a hardware capture. Neither
`Consume` nor the CLI revalidates upstream numerical/timing stages. Original
corpus schemas, records, kernel sources and existing advice rules stay unchanged.
