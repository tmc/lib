# NCCL testing parity

Objective: replicate nccl-tests-style testing in a Go-native x/ package and
prove the resulting behavior against nccl-tests on the DGX.

Reference: NVIDIA/nccl-tests b4d5beebca8a76cf01335f724d154b9b9d394d96.

Required evidence before completion:

- Go testing and benchmarking entry points, reusable runner, deterministic
  independent host oracle, size sweeps, correctness counts, CUDA-event timing,
  algorithm/bus bandwidth calculations, and machine-readable receipts.
- Host collective paths: all_reduce, all_gather, broadcast, reduce_scatter,
  reduce, alltoall, alltoallv, scatter, gather, sendrecv, hypercube. Match upstream
  count alignment, root, reduction, datatype, in/out-of-place semantics, warmup,
  iterations, aggregation, and report which advanced upstream modes are covered.
- Exercise the correctness checker with corrupted data and an omitted operation;
  zero comparisons or skipped GPU tests cannot prove correctness.
- Run upstream and Go with matching effective cases and exact library/source
  provenance on the DGX. Keep raw output and process exit status. Separate
  correctness agreement from timing differences and from unsupported cases.
- Record actual GPUs/ranks. The inspected DGX has one GB10, so physical multi-GPU
  transport cannot be inferred from its single-rank results. Do not disguise
  that boundary with repeated ranks on one device.
- Retain comparison artifacts, locally tested documented commands, and an
  explicit coverage report. Do not mark complete from local/mock tests alone.

Current worktree: ~/tmp/nccl-go-testing/worktree, branch codex/nccl-go-testing.
The shared main tree has unrelated work and must not be reset or overwritten.

Progress 2026-09-05:
- All eleven default host paths implemented for float32, with independent
  oracle, typed layouts, source-matched grouping, and native testing/benchmarking.
- Added safe nccl.InitAll slice output; no use of the old single-slot constructor.
- DGX matrix-02: 129 checked cases, 1,431,384 comparisons, zero errors, all five
  reductions. matrix-01 retained its hypercube empty-column parser failure.
- matrix-03 revalidates stronger hash-derived rank payloads and source-matched
  single-GPU grouping. See ~/tmp/nccl-go-testing/receipts on the DGX/local host.
- Do not declare the goal complete yet: complete the datatype/advanced-mode
  coverage audit, run physical Go tests/negative control/benchmarks, retain exact
  final source provenance, and explicitly resolve the one-GPU coverage boundary.

Milestone closure:
- matrix-03 passed 129 checked float32 cases with stronger rank payloads.
- matrix-04 passed the 12 gather/scatter cases after matching native NCCL entry
  points. Canonical coverage stays 129, not 141 (the rows supersede earlier ones).
- Physical Go suite and all 11 fixed-count benchmarks pass. Missing-collective
  GPU control rejects 16,384 of 16,384 values. Final source checksums match DGX.
- See DGX.md for exact scope, observed timing differences, and remaining work.
- No background GPU run remains live after the retained test/benchmark command
  completed. Requalify host process/GPU state before any next launch.

Aggregation milestone:
- Added source-matched outer NCCL groups for timed aggregation, operation-count
  reporting, default-one semantics, overflow validation, and group failure tests.
- Comparison parser now verifies native printed controls and matching Go counts.
- DGX aggregation-02: all 11 operations, 19 checked modes, aggregation 4, 80 timed
  operations/case, 622,592 comparisons, zero errors. Default aggregation regression
  and TestAggregation also pass. Source/binary hashes and all raw logs retained.
- Local race tests and vet pass. Goal remains active; next major gap is datatype
  coverage beyond float32 and pre-multiplied sum, followed by the advanced-mode
  audit and explicit disposition of the single-GPU hardware boundary.

Pre-multiplied-sum milestone:
- Added PreMulSum/mulsum to the case API and command comparison matrix.
- Matched upstream alternating scalar values and per-call custom-op lifetime;
  pinned the host-immediate scalar across its integer-address FFI boundary.
- DGX: 18 unaggregated plus 6 aggregated checked cases, 396,336 comparisons,
  zero errors. The pinned aggregated receipt supersedes its earlier version.
- Native Go tests for all three affected collectives, aggregation, and the
  omitted-operation control pass. Local race tests/vet and example checks pass.
- No goal completion claim: datatype coverage beyond float32, advanced-mode
  disposition, and the single-GPU proof boundary remain. See DGX.md.

Reference-control audit:
- Added COVERAGE.md, enumerating every option in the pinned reference's option
  table and the all-to-all-v environment controls. Missing modes stay open.
- Found and fixed inherited reference-only overrides: ncclcompare now rejects
  NCCL_TESTS_* before launching children and checks the returned Go Case exactly.
- Local race tests and vet pass. DGX environment-01 passes three 64-KiB checked
  cases; environment-rejection.log proves a spread override exits before launch.
  Source and executable hashes retained in environment-{source,binaries}.sha256.
- Datatype implementation remains the next major gap; full equivalence is still
  unproven, and this audit does not reduce the objective to existing coverage.

Float64 milestone:
- Added DataType, Float64, element-width-aware layouts/bandwidth/offsets, raw
  transfer encodings, double collective arguments and host-immediate scalars,
  and Check64 with double host arithmetic. Both commands accept -datatype.
- Double payloads include bits below float32 precision. Known-bit encoding,
  rank routing, width/alignment, and float32-narrowing negative tests pass.
- Fixed a checker false-pass risk: nonfinite expected values now return an error
  instead of producing an infinite tolerance. Large-rank overflowing products
  still need bounded payloads before those cases can be covered.
- DGX double-01 passed 147 cases and 815,556 comparisons. The final follow-up
  snapshot passed 49 aggregated double cases (802,816 comparisons) and 49
  float32 regression cases (1,605,632 comparisons), all with zero wrong values.
- Native Go tests passed 38 collective modes and both missing-operation controls;
  all 22 fixed-count benchmarks passed. Local race tests and vet pass. Final
  harness hashes match the local worktree; no remote test process remains.
- Goal remains active. Integer, half, bfloat16, FP8, the remaining controls in
  COVERAGE.md, and physical multi-GPU proof remain open.

Integer implementation and active validation:
- Added signed/unsigned 8-, 32-, and 64-bit datatypes, native scalar widths,
  bit-exact CheckBytes, width-wrapped reductions, signed ordering/division,
  full-width rank payloads, and complemented expected-output poisoning.
- Fixed three-rank arithmetic vectors, low-bit corruption beyond float64's
  exact range, encoding, byte-length validation, and offset poisoning tests
  pass. Local race tests and vet pass.
- The DGX is running six complete matrices sequentially, in this order:
  int8, uint8, int32, uint32, int64, uint64. Output directories are
  receipts/integer-<datatype>-01. The active local exec handle is 47419;
  poll that handle or verify its remote process before treating it as stopped.
  Do not launch a duplicate matrix or replace its executables while it runs.
- int8 completed: 147 cases, 6,524,448 comparisons, zero errors, 225 successful
  child processes. The remaining matrices were still running at this checkpoint.
- integer-source.sha256 and integer-binaries.sha256 identify the running
  snapshot. integer-harness.tar.gz archives its source; all 18 Go-file hashes
  have been verified against the archive. Receipts are copied locally.
- Local work after that snapshot bounds floating products using four
  magnitude-bearing ranks followed by variable signs, and adds comparator
  -root support. These changes pass local tests but are not in the active DGX
  executable. Keep the source snapshots distinct in reporting.
- After the matrix sequence finishes: copy and verify all completed receipts;
  sync/rebuild the latest source; run NCCLTEST_CUDA=1 Go tests and all 88 fixed
  benchmarks (allow 300 seconds per process); validate integer aggregation
  against the reference and the changed floating/root paths; retain final hashes.
- Multi-GPU host availability has been asked asynchronously. No answer is
  assumed; independent single-GPU implementation/validation continues.

Integer milestone closure and codec preparation:
- Exec 47419 completed successfully: all six integer matrices passed, totaling
  882 cases, 17,942,232 comparisons, and 1,350 successful child processes.
  All candidate attempt hashes match integer-binaries.sha256.
- Exec 55212 also completed successfully on the latest source: 152 physical
  collective modes, eight omitted-operation controls, and 88 benchmarks passed.
  Integer aggregation added 36 cases and 2,162,688 comparisons with no errors;
  twelve floating product regressions and invalid-root rejection passed too.
- integer-final-source.sha256 matches the local Go sources. Both source archives,
  executable manifests, raw logs, and verification JSON files are retained.
  No validation or GPU compute process remains live; do not poll either old
  exec handle or rerun their matrices without a new reason.
- Added fixed nonzero-root and width-dependent integer routing samples, plus
  rank-permutation controls for all six integer types. Local race tests/vet pass.
- Added private half/bfloat16/FP8 codecs with exhaustive finite round trips and
  independent CUDA encoder/decoder checks: 498,146 distinct inputs, 1,992,584
  checks in each direction. Local CUDA 13.3/Clang and DGX CUDA 13.0/GCC fixtures
  produced byte-identical output. Header dependencies and raw corpus are retained.
- The four small floating datatypes are NOT yet selectors in Case or commands.
  SMALLFLOAT.md records the integration plan and exact-arithmetic input design.
  Next: verify native/reference average scaling, wire the codecs into buffers,
  layouts, scalars, and the oracle, then run matched physical datatype matrices.
- Goal remains active. Other controls in COVERAGE.md and physical multi-GPU
  proof remain open. The multi-GPU host question has no answer at this checkpoint.

Coordination hold (2026-09-05):
- This session owns codex/nccl-go-testing and this worktree. All previous remote
  runs completed. Read-only process and GPU compute-app checks returned empty.
- Explicitly released this task's DGX use to coordinating session 54B33A62 for
  its bounded qualification window. Hold all GPU launches until that session
  sends a release/completion callback. Idle hardware alone does not lift HOLD.
- The coordinator accepted a ten-minute correctness-only window and requested
  a hold on GPU and heavy CPU/I/O work until its explicit RELEASE/HOLD callback.
  The deadline alone does not authorize resuming work.
- Session B4C3EC3E requested a subsequent correctness smoke slot. Referred it
  to 54B33A62 for explicit handoff; no overlapping slot was authorized here.
  This task remains on hold through that coordinated slot and awaits cleanup
  RELEASE from B4C3EC3E as well before resuming heavy work.
- 54B33A62 sent RELEASE: tests exited, final inventory was empty, and locks
  were released. It named B4C3EC3E as next requested smoke owner. Forwarded
  that release and confirmed this task's continued hold to B4C3EC3E; its
  cleanup RELEASE is still required before resuming heavy work.
- B4C3EC3E subsequently sent cleanup RELEASE: its runner exited successfully,
  locks were released, and final compute inventories were empty. It reports
  no further remote work queued. Both coordination holds are now cleared.
  Acquire HOST-shared then UUID-exclusive locks and check fresh process/GPU
  inventory before any new DGX launch; these callbacks are not fresh inventory.
  Peer smoke evidence: /Users/tmc/tmp/cuda-benchmark-fix/smoke-telemetry.log
  (reported by the peer, not independently reviewed here).
- B4C3EC3E requested a fresh roughly five-second correctness smoke after
  rebasing onto updated main. No new remote work launched from this task.
  Confirmed release of this task's use and renewed the GPU/heavy CPU/I/O hold
  until its next cleanup RELEASE. B4C3EC3E must also obtain 54B33A62's
  confirmation before launch; this confirmation covers only this task.
- B4C3EC3E sent final smoke RELEASE for its rebased binary at 7b6b704 on
  9eb4e1d: runner exited 0, locks released, fresh final compute inventory
  empty, and no remote work queued. The renewed hold is cleared. Evidence
  reported by the peer: /Users/tmc/tmp/cuda-benchmark-fix/smoke-rebased.log;
  not independently reviewed here and not evidence for NCCL equivalence.
  Fresh locks and inventory checks remain required before our next launch.
- New slot: explicitly handed off this lane's spark-31db use to 2379D0C3 for
  a bounded ten-minute correctness window including cleanup. No new remote
  work has launched from this task and none is queued. Hold remote work until
  its RELEASE/HOLD callback; timeout alone does not lift the hold. Requester
  must use HOST-shared then UUID-exclusive locks, an 8GiB floor, fresh
  inventories, stop on conflicts, and retain locks through cleanup.
- 2379D0C3 sent RELEASE: both diagnostic runners exited, no owned descendants
  or cleanup errors, post-run and fresh locked compute inventories empty,
  locks released, and no remote work queued. This remote-work hold is cleared.
  Peer reports application failures (nonfinite logit and repeat failure), with
  zero initcheck sanitizer errors; this is not a correctness pass. Cleanup
  evidence reported at /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/single-token-candidate04/release01.json
  and plain01/initcheck01 receipts.tar.gz, not independently reviewed here.
  Acquire fresh locks and check inventory before this task's next remote run.
- New handoff to 2379D0C3: bounded five-minute correctness slot for frozen
  diagnostic b25e5db35a49f717, one run max180s plus cleanup within the window.
  No remote work launched or queued from this task. Hold remote work until
  its cleanup RELEASE/HOLD callback; timeout alone does not lift this hold.
  Shared HOST then exclusive UUID locks, 8GiB floor, fresh inventories,
  stop-on-conflict checks, and locks through cleanup remain required.
- 2379D0C3 released the layer-snapshot slot: reports exit 0, no live owned
  descendants or cleanup errors, fresh locked inventory empty, locks released,
  and no queued remote work. This hold is cleared; any next experiment needs
  a new slot. Its reported 87 identical snapshot pairs and final logits apply
  only with instrumentation, not uninstrumented parity or NCCL validation.
  Peer receipts (not independently reviewed): release01.json and
  plain01/receipts.tar.gz under
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/single-token-layer-probe01/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot, up to four
  predefined sparse-barrier controls with frozen binary e41bcd39dc4305ac,
  max120s each plus cleanup within the overall window. No new remote work
  launched or queued from this task. Hold remote work until its cleanup
  RELEASE/HOLD callback; timeout alone does not lift the hold. Shared HOST
  then exclusive UUID locks, 8GiB floor, fresh inventories, stop on conflict,
  and locks through cleanup remain required.
- 2379D0C3 released the barrier-control slot: reports all four runners exited
  and cleaned up, fresh locked inventory empty, locks released, and no queued
  remote work. This hold is cleared. None/embedding/layers reportedly passed
  and midpoint failed; numerical diagnosis remains open, with no timing claim.
  Peer evidence (not independently reviewed): release01.json and comparison01.json
  under /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/single-token-barrier-probe01/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot for three
  scratch-initialization controls (none/zero/NaN), frozen be984c1713a937b6,
  max120s per run with cleanup inside the window. Independent test review
  must pass before launch. No remote work launched or queued from this task.
  Hold remote work until its cleanup RELEASE/HOLD callback; timeout alone
  does not lift the hold. Shared HOST then exclusive UUID locks before
  transfers, 8GiB floor, fresh inventories, stop on conflict, and locks through
  cleanup remain required.
- 2379D0C3 released the scratch-control slot: reports all three controls ended,
  no remaining owned processes or cleanup errors, fresh locked inventory empty,
  locks released, and no queued remote jobs. This hold is cleared. None and
  zero reportedly failed while NaN passed; no root-cause or timing claim.
  Peer evidence (not independently reviewed): comparison01.json and release01.json
  under /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/single-token-scratch-probe01/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness slot for unchanged
  diagnostic binary fb75ca7d under racecheck then synccheck, each max120s with
  cleanup inside the overall window. No remote work launched or queued from
  this task. Hold remote work until its cleanup RELEASE/HOLD callback; timeout
  alone does not lift the hold. Shared HOST then exclusive UUID locks before
  transfer, 8GiB floor, fresh inventories, stop on conflict, and locks through
  cleanup remain required.
- 2379D0C3 released the racecheck/synccheck slot: reports both runs finished,
  all owned processes gone, no cleanup errors, fresh locked inventory empty,
  locks released, and no queued remote jobs. This hold is cleared. Both
  applications reportedly failed despite zero reported sanitizer hazards/errors;
  this is not a correctness pass. No timing claim.
  Peer evidence (not independently reviewed): sanitizer-comparison02.json and
  release02.json under
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/single-token-candidate04/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness slot for one
  queued-device-snapshot probe, binary 14820916e1fda6fc, max120s plus cleanup
  within the overall window. No remote work launched or queued from this task.
  Hold remote work until its cleanup RELEASE/HOLD callback; timeout alone does
  not lift the hold. Shared HOST then exclusive UUID locks before transfer,
  8GiB floor, fresh inventories, stop on conflict, and locks through cleanup
  remain required.
- 2379D0C3 released the queued-snapshot slot: reports exit 0, no live owned
  processes or cleanup errors, fresh locked inventory empty, locks released,
  and no queued remote jobs. This hold is cleared. The reported 87 identical
  snapshot pairs and final arrays are diagnostic evidence only, with no timing
  or uninstrumented-parity claim.
  Peer evidence (not independently reviewed): comparison01.json and release01.json
  under /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/single-token-queued-probe01/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness slot for one
  current-integrated-runtime Reset probe, frozen c1819f3f6b1e139e, max120s
  plus cleanup within the overall window. No remote work launched or queued
  from this task. Hold remote work until its cleanup RELEASE/HOLD callback;
  timeout alone does not lift the hold. Shared HOST then exclusive UUID locks
  before transfer, 8GiB floor, fresh inventories, stop on conflict, and locks
  through cleanup remain required.
- 2379D0C3 released the current-runtime Reset slot: reports exit 0, no live
  owned processes or cleanup errors, fresh locked inventory empty, locks
  released, and no queued remote jobs. This hold is cleared. Reported exact
  finite logit arrays are from one short diagnostic, not full parity.
  Peer evidence (not independently reviewed): release01.json and RESULT-01.md
  under /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/current-single-token01/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute slot for ONE current-runtime
  count30 capture, frozen binary
  c1819f3f6b1e139e8de3b5129421e3d4e078fd6d962e7a4b5fbf17771be21466,
  max300s application run with transfer and cleanup inside the overall window.
  Independent runner review must pass before launch. No remote work launched
  or queued from this task. Hold remote work until its cleanup RELEASE/HOLD
  callback; timeout alone does not lift the hold. HOST shared then UUID
  exclusive before transfers, fresh inventories, 8GiB floor, stop on conflict,
  and locks through cleanup remain required. Handoff covers only this lane.
- 2379D0C3 released the count30 slot: reports application exit 0 and all required
  states captured, no remaining owned process-group members, no cleanup errors,
  and empty post-run compute inventory. Fresh both-lock inventory at
  2026-09-06T00:17:34Z reportedly exited 0 and was empty; locks released and no
  remote work queued. This hold is cleared. Numerical comparison remains
  pending locally; there is no timing claim.
  Peer evidence (not independently reviewed): release01.json and capture01
  under /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/current-count30-01/.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute slot for one current-runtime
  token0/pos0 layer endpoint probe, frozen binary
  1b5f31d11f300df8e69c42683f9032ff47101804fda324b678b6ebdc69f53829,
  max120s test run with transfers and cleanup inside the window. Independent
  review must ACCEPT before launch. No remote work launched or queued from
  this task. Hold remote work until its cleanup RELEASE/HOLD callback; timeout
  alone does not lift the hold. Shared HOST then exclusive UUID locks, fresh
  inventories, 8GiB floor, stop on conflict, and locks through cleanup remain
  required. Handoff covers only this lane.
- 2379D0C3 released the layer endpoint slot: reports application exit 0,
  owned process/group empty, no cleanup errors, and fresh both-lock inventory
  at 2026-09-06T00:23:09Z exited 0 and empty. Locks released and no remote
  work queued. This hold is cleared. Endpoint comparison remains local;
  no timing claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/current-layer-probe01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness-development slot
  to compile/link only the layer0 oracle observer (current source SHA recorded
  in its oracle-build-plan.json, all 16 original object/source hashes checked),
  max120s, then ONE candidate layer0 diagnostic
  bf52ae56f3967dd366f71ce27c1d4299d0abc77699ceb8b84ba57c79dd1f71b5,
  max120s. Transfers and cleanup inside the window; no oracle model run.
  Independent runner review must ACCEPT before work. No remote work launched
  or queued from this task. Hold remote work until cleanup RELEASE/HOLD;
  timeout alone does not lift the hold. HOST shared then UUID exclusive before
  work, fresh inventories, 8GiB floor, stop on conflict, and locks through
  cleanup remain required. Handoff covers only this lane.
- 2379D0C3 released the observer-build/candidate slot: reports compile/link
  exit 0, all 16 pinned inputs unchanged, compiler and candidate process groups
  empty, no cleanup errors, and fresh both-lock inventory at
  2026-09-06T00:34:39Z exited 0 and empty. Locks released, no remote work
  queued, and no oracle model run occurred. This hold is cleared. Candidate
  probe failed before its first stage due to a reported instrumentation
  validator defect; no parity inference. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/current-layer0-stages01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot for ONE layer0
  oracle diagnostic, frozen
  00bc5e3ef339467c6eac18cf4e0120b12c7d261bc093ace8156480d25a6cfa3d,
  max300s, then ONE corrected candidate layer0 probe
  588c59e756d7fccd5dfdedf302bf92ccb60c04e15d18636d9a3822f6dbce56c6,
  max120s. Transfers and cleanup inside the overall window. Independent runner
  review must ACCEPT before work. No remote work launched or queued from this
  task. Hold remote work until cleanup RELEASE/HOLD; timeout alone does not
  lift the hold. Shared HOST then exclusive UUID before work, 8GiB floor,
  fresh inventories, stop on conflict, and locks through cleanup remain
  required. Handoff covers only this lane.
- 2379D0C3 released the oracle/corrected-candidate slot: reports both exited 0,
  candidate captured 19 finite stages, owned process groups empty, no cleanup
  errors, and fresh both-lock inventory at 2026-09-06T00:39:34Z exited 0 and
  empty. Locks released and no remote work queued. This hold is cleared.
  Local comparison remains pending; no timing claim. Peer receipt (not
  independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/current-layer0-stages01/release02.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot for three
  frozen raw-prefix controls, max120s each with transfers/cleanup inside the
  window: old=6329e381256f48cd2ad5f28c6780f5e03d2f9594496d057ed993079e8dd010a1,
  score-mutant=8c9981eebf30f5c08c84525db5562b1708d29489eacf04442ee1d95c9a620a78,
  fixed=dc94c94e9e91b1092690a05ac7e920b18b1b1df5c4945e7ca06ae8bde9c75412.
  Independent review must ACCEPT before work. No remote work launched or
  queued from this task. Hold remote work until cleanup RELEASE/HOLD; timeout
  alone does not lift the hold. HOST shared then UUID exclusive before
  transfers/work, 8GiB floor, fresh inventories, stop on conflict, and locks
  through cleanup remain required. Handoff covers only this lane.
- 2379D0C3 released the raw-prefix control slot: reports expected numerical
  rejections for old and score-mutant controls, fixed regression/history/stages
  passed with exit 0, all three owned groups empty, and no cleanup errors.
  Fresh both-lock inventory at 2026-09-06T00:47:55Z reportedly exited 0 and
  was empty; locks released and no remote work queued. This hold is cleared.
  Local stage comparison remains pending; no timing claim. Peer receipt
  (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/raw-prefix-fix01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness slot for ONE
  teacher-forced output projection probe, binary
  98e09b8b0764db9e210c9b1b9980b3186f44da856b1adf29f6c6cde14c4a5e03,
  max120s with transfers and cleanup inside the window. Independent review
  must ACCEPT before work. No remote work launched or queued from this task.
  Hold remote work until cleanup RELEASE/HOLD; timeout alone does not lift
  the hold. HOST shared then UUID exclusive, 8GiB floor, fresh inventories,
  stop on conflict, and locks through cleanup remain required. Handoff covers
  only this lane.
- 2379D0C3 released the teacher-projection slot: reports exit 0, forced-heads
  SHA matching the pinned oracle fixture, owned process/group empty, and no
  cleanup errors. Fresh both-lock inventory at 2026-09-06T00:53:16Z reportedly
  exited 0 and was empty; locks released and no remote work queued. This hold
  is cleared. No timing claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/outproj-teacher01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness slot for one
  quantizer compiler-mode probe, frozen binary
  f4549af3d50589a71c8768311cf026d7405f1a060dd0277cc37e20a93aaeef17,
  max120s including two small NVRTC compilations and synthetic launches.
  No checkpoint evaluation; transfers and cleanup inside the window.
  Independent runner review must ACCEPT before work. No remote work launched
  or queued from this task. Hold remote work until cleanup RELEASE/HOLD;
  timeout alone does not lift the hold. HOST shared then UUID exclusive before
  transfers, fresh inventories, 8GiB floor, stop on conflict, and locks through
  cleanup remain required. Handoff covers only this lane.
- 2379D0C3 released the quantizer compiler-mode slot: reports exit 0 and both
  modes passed, owned process/group empty, and no cleanup errors. Fresh
  both-lock inventory at 2026-09-06T00:58:33Z reportedly exited 0 and was empty;
  locks released and no remote work queued. This hold is cleared. No timing
  claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/quant-modes01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness window to compile/link
  the standalone pinned oracle quantizer, source manifest SHA
  6c4fbc091b656e2cae6241b2b568c34cc4fad0dbec57205aca7c3ea50f955698,
  max120s, then one synthetic launch on retained 131072-byte oracle-head input,
  max120s. No checkpoint evaluation. Independent ACCEPT is mandatory before
  transfers or work; transfers/cleanup inside the window. No remote work
  launched or queued from this task. Hold remote work until cleanup
  RELEASE/HOLD; timeout alone does not lift the hold. HOST shared then UUID
  exclusive, fresh inventories, 8GiB floor, stop on conflict, and locks through
  cleanup remain required. Handoff covers only this lane.
- 2379D0C3 released the standalone oracle-quantizer slot: reports compile and
  synthetic launch exited 0, no remaining owned processes or cleanup errors,
  fresh locked compute inventory empty, locks released, and no queued remote
  work. This hold is cleared. Peer reports NVCC output matched NVRTC fast mode
  byte-for-byte while default differed in 48 scales and 72 codes; diagnostic
  only, no timing claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/oracle-quantizer-run01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness window for binary
  51e90cd057f6675075cb1ae8e16db4f7b7fb43815622432158f8e094eea709e8.
  First synthetic exact-oracle quantizer and existing Q80 quantization/matvec
  tests, max120s; only if all pass, one teacher-forced layer0 projection
  checkpoint probe, max120s. No tolerance changes. Final runner review must
  ACCEPT before work. No remote work launched or queued from this task.
  Hold remote work until cleanup RELEASE/HOLD; timeout alone does not lift
  the hold. Shared HOST then exclusive UUID, 8GiB floor, fresh inventories,
  stop on conflict or first defect, and locks through cleanup remain required;
  transfers and cleanup inside ten minutes. Handoff covers only this lane.
- 2379D0C3 released the Q80/teacher-projection slot: reports synthetic gates
  passed and projection probe exited 0, both owned process groups cleaned,
  no live members or cleanup errors, fresh locked compute inventory empty,
  locks released, and no queued remote jobs. This hold is cleared. OutA
  reportedly matches exactly, but OutB retains a strict mismatch (max0.000651);
  this is not a full correctness pass. No timing claim. Peer receipt (not
  independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/outproj-teacher02/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness window for ONE
  test-only teacher projection route probe, frozen binary
  923168433e3962330751091e0583561c5652e97d7a8db7e5ea803e469b550b9d,
  max120s with transfers and cleanup inside the window. Probe OutB dispatcher
  change only; production and tolerances unchanged. Independent runner review
  must ACCEPT before work. No remote work launched or queued from this task.
  Hold remote work until cleanup RELEASE/HOLD; timeout alone does not lift
  the hold. Shared HOST then exclusive UUID, fresh inventories, 8GiB floor,
  stop on conflict, and locks through cleanup remain required. Handoff covers
  only this lane.
- 2379D0C3 released the projection-route slot: reports probe exit 0, owned
  process group empty, no cleanup errors, fresh locked compute inventory empty,
  locks released, and no queued remote work. This hold is cleared. OutB
  reportedly matches exactly with the test-only prefill dispatcher; no production
  edit or timing claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/outproj-route01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness window for binary
  cf8d2f0f9160e2fb36f86b88f3723ae16e2b17103084639eaed13f50bcfc4daa.
  First standalone quantizer regression and existing Q80 tests, max120s; only
  if all PASS, one production prefill-layer endpoint probe (token0,pos0,two
  Reset iterations), max120s. No teacher inputs or tolerance changes. Final
  runner review must ACCEPT before work. No remote work launched or queued
  from this task. Hold remote work until cleanup RELEASE/HOLD; timeout alone
  does not lift the hold. Shared HOST then exclusive UUID, 8GiB floor, fresh
  inventories, stop on conflict or first defect, and locks through cleanup
  remain required; transfers/cleanup inside the window. Handoff covers only
  this lane.
- 2379D0C3 released the prefill endpoint slot: reports synthetic Q80 tests
  passed and capture exited 0, but the numerical finalizer exited 1 at layer0
  attention HC (max0.0004102, 1435 strict cells). All 87 HC pairs reportedly
  repeat exactly; stable mismatch is not qualification. Owned cleanup empty,
  no errors, fresh locked compute inventory empty, locks released, and no
  queued remote jobs. This hold is cleared. No timing claim. Peer receipt
  reported as current-layer-probe02/release01.json (not independently reviewed).
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit lane release to B4C3EC3E: at-most-two-minute correctness window
  at f8245e6, frozen llama-nvidia.test SHA256
  06f984e0c166809abf48b5e406048d5b32f5b614246e513ea0543ea112ea5432,
  covering transfer, model/serving correctness, module-lifetime test, and cleanup.
  No remote work launched or queued from this task. Hold remote work until its
  cleanup RELEASE/HOLD callback; timeout alone does not lift the hold.
  B4C3EC3E must obtain 54B33A62 confirmation, use HOST-shared then UUID-exclusive
  locks and fresh inventories, stop on conflict, and retain locks through cleanup.
  Identified 2379D0C3 as the latest additional requester, whose slot was released;
  no additional active owner is known here. This release covers only this lane.
- B4C3EC3E sent cleanup RELEASE for f8245e6: reports correctness suite exit 0,
  HOST-shared then UUID-exclusive locks held through transfer/tests/cleanup,
  final compute and scoped process inventories empty at 2026-09-06T01:57:18Z,
  lock shell exited, and no queued remote work. This hold is cleared.
  Peer reports model, serving, and module-lifetime checks passed; no timing
  comparison. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/llama-nvidia-validation/smoke.log.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot to compile/link
  standalone pinned raw-attention kernel, source freeze
  f1d3c42f9291647fc0e9d0626b548031cd7ddd870b41d0733c104eecae6b1f6a,
  max120s, then ONE synthetic launch on retained Q/KV/sink fixtures, max120s,
  with mandatory exact retained-head comparison. Runner freeze 69f0a41e;
  independent review must ACCEPT before transfers/work. No checkpoint
  evaluation or numerical edits. No remote work launched or queued from this
  task. Hold remote work until cleanup RELEASE/HOLD; timeout alone does not
  lift the hold. HOST shared then UUID exclusive before transfers, 8GiB floor,
  fresh inventories, stop on conflict or first defect, and locks through cleanup
  remain required; all work/cleanup within ten minutes. Covers only this lane.
- 2379D0C3 released the raw-attention slot: reports host C compilation failed
  with misleading-indentation warnings treated as errors, before NVCC or GPU
  launch; build/overall exit 1 retained without retry. No remaining owned
  processes or cleanup errors. Fresh locked compute/scoped-process inventory
  reportedly empty at 2026-09-06T02:13:13.820646Z, probe exited, locks released,
  and no queued remote work. This hold is cleared; a repair needs a new slot.
  Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/oracle-raw-attention-run01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot for source
  freeze 4b0a74958ab9b262fe2b7f2e0329576187804ae14231ba771b45f68c16ee56e6,
  runner 9a2e1637. Compile max120s, then ONE synthetic raw-attention launch
  max120s with mandatory exact heads check; transfers/cleanup within ten minutes.
  Peer reports independent delta review ACCEPT for host-wrapper formatting only;
  prior build failure retained, no checkpoint evaluation. No remote work launched
  or queued from this task. Hold remote work until cleanup RELEASE/HOLD; timeout
  alone does not lift the hold. HOST shared then UUID exclusive before transfers,
  8GiB floor, fresh inventories, stop on conflict or first defect, and locks through
  cleanup remain required. Covers only this lane.
- 2379D0C3 released raw-oracle run02: reports compile/run/mandatory comparison
  exit 0, 32768 heads byte-exact, no remaining owned processes or cleanup
  errors, fresh locked compute/scoped inventory empty at
  2026-09-06T02:17:58.518765Z, and locks released. No model equivalence claim.
  Peer receipt (not independently reviewed):
  oracle-provenance/oracle-raw-attention-run02/release01.json.
- NEW separate handoff to 2379D0C3: five-minute correctness slot for reviewed
  raw-arithmetic-modes02 runner manifest a033d0b0, binary
  3eed51b291ddf27940dc08bc676738d087ebfc34411d3a8d11de124d0f051a18,
  ONE synthetic test max120s comparing default/scaling/fast modes. No checkpoint
  or production edit; transfers/cleanup within the window. No remote work
  launched or queued from this task. Hold remote work until cleanup RELEASE/HOLD;
  timeout alone does not lift the hold. HOST shared then UUID exclusive, 8GiB
  floor, fresh inventories, stop on conflict, and locks through cleanup remain
  required. Handoff covers only this lane.
- 2379D0C3 released raw-arithmetic-modes02: reports valid captures and exact
  Qcur/KVcur in all three modes, but heads differ from the oracle in every mode;
  diagnostic only, no numerical acceptance. Process cleanup empty, no errors,
  fresh locked compute/scoped inventory empty at 2026-09-06T02:19:36.419767Z,
  locks released, and no remote work queued. This hold is cleared; no new slot
  requested. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/raw-arithmetic-modes02/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness slot for instrumented
  standalone oracle source freeze
  df87622b22ee6284dd8e8765d5497d98d3be8b452aa9afed4c7067b28cd07bd1,
  runner 18c05ccd. Compile max120s then ONE synthetic launch max120s;
  transfers/cleanup inside the window. Adds 768-byte score/exp/denom observations;
  arithmetic/fixtures/flags unchanged, mandatory exact 32768-head comparison
  before observations qualify. Independent runner ACCEPT required before work;
  no checkpoint or production edits. No remote work launched or queued here.
  Hold remote work until cleanup RELEASE/HOLD; timeout alone does not lift
  the hold. HOST shared then UUID exclusive, 8GiB floor, fresh inventories,
  stop on conflict or first defect, and locks through cleanup remain required.
  Handoff covers only this lane.
- 2379D0C3 released the observer slot: reports build/run/comparison exit 0,
  original 32768 heads byte-exact and 192 observations qualified. Cleanup had
  no remaining owned processes or errors; fresh locked compute/scoped inventory
  empty at 2026-09-06T02:30:30.322161Z, locks released, and no remote work
  queued. This hold is cleared. Peer reports scaled-score differences requiring
  local analysis; no model or timing claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/oracle-raw-observer-run01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: five-minute correctness slot for reviewed
  raw-dynamic-scale01 runner manifest cc322ef2, binary
  56ba77f50053f97417f11af537e71c27eeceeb3e965d29809fc39e05fbcd0477.
  ONE synthetic test max120s, three runtime-rsqrt modes default/fast/intrinsic,
  six tiny launches total including coefficient observers. No checkpoint or
  production edit; measured coefficients, no fitted expected constant. Peer
  reports source/runner ACCEPT. No remote work launched or queued from this
  task. Hold remote work until cleanup RELEASE/HOLD; timeout alone does not
  lift the hold. HOST shared then UUID exclusive before transfer, 8GiB floor,
  fresh inventories, stop on conflict, and locks through cleanup remain required;
  transfer/work/cleanup within five minutes. Handoff covers only this lane.
- 2379D0C3 released the dynamic-scale slot: reports root/three modes passed,
  capture exit 0, no remaining owned processes or cleanup errors, fresh locked
  compute/scoped inventory empty at 2026-09-06T02:41:09.466146Z, locks released,
  and no queued remote work. This hold is cleared. Peer reports runtime scale
  plus narrow exp/div intrinsics matched 32768 oracle heads and 64 scores
  byte-exact; synthetic diagnosis only, not production acceptance. Peer receipt
  (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/raw-dynamic-scale01/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- New explicit handoff to 2379D0C3: ten-minute correctness window for reviewed
  probe03 freeze 7d256795, binary
  1e4ef9b8aae54703b9d11ff048ad3dbbcb8dfba0f1c124108bca1f1ca791152b.
  Six synthetic regression roots max120s; ONLY if all PASS, one normal prefill
  token0/pos0 two-Reset endpoint probe max120s with unchanged 86x2 strict HC
  comparisons. No teacher inputs, tolerance edits, or timing claim. Peer reports
  independent build/runner ACCEPT. No remote work launched or queued here.
  Hold remote work until cleanup RELEASE/HOLD; timeout alone does not lift the
  hold. Other coordinated lane release required before work. HOST shared then
  UUID exclusive before transfers, 8GiB floor, fresh inventories, stop on conflict
  or first failure, and locks through cleanup remain required; all work/transfers/
  cleanup inside ten minutes. Handoff covers only this lane.
- 2379D0C3 released probe03: reports campaign exit 1 after synthetic/capture
  exit 0 and numerical finalizer exit 1, without retries. Runner groups reaped;
  fresh locked compute/scoped-process inventory empty at
  2026-09-06T03:15:40.074193Z, probe SSH exited and locks released. No remote
  work queued. This hold is cleared. Peer reports first two layers satisfy
  strict endpoints, with first remaining failure at layer2 attention; this is
  not full qualification. No timing claim. Peer receipt (not independently reviewed):
  /Users/tmc/tmp/ds4-lib-workflow/evidence/oracle-provenance/current-layer-probe03/release01.json.
  Fresh locks and inventory checks remain required before our next remote run.
- 2026-09-09: fresh DGX SSH attempt rejected authentication
  (Permission denied publickey,password), before any remote command executed.
  No remote launch occurred. Linux ARM64 CGO-disabled cross-builds of nccltest,
  ncclcompare, and the package test binary passed. Frozen binaries and source
  hashes are in /Users/tmc/tmp/nccl-go-testing/smallfloat-20260909/manifest.json.
  These are build evidence only; restore SSH access before native validation.
- Local continuation: added four-format copy/variable-traffic routing and
  permutation controls, linear size sweeps, comparator sweep flags, and complete
  workload preflight before receipt creation or native execution. Tests/vet pass.
  Seven actual invalid CLI invocations exited 2 without creating output or
  launching marker executables; a valid invocation triggered the marker and
  retained exit 1, proving the launch detector works. Evidence:
  /Users/tmc/tmp/nccl-go-testing/preflight-20260909/result.json.
  This is command-validation evidence, not GPU equivalence. Earlier frozen
  smallfloat-20260909 binaries predate these changes and must be rebuilt before
  validating current source. Requested restored DGX authentication from user;
  meaningful local implementation work remains possible.
- SSH restored on fresh check. Host remains spark-31db/GB10; compute inventory
  was empty, but no workload launched. 2379D0C3 now explicitly reports work
  stopped and transferred to 6E6A46AB. Requested a new ten-minute correctness
  slot from that active DS4 owner; await explicit handoff before transfers or
  launch. Latest binaries rebuilt in ~/tmp/nccl-go-testing/native-20260909/.
- 6E6A46AB explicitly released its DS4 lane for our ten-minute window and is
  holding until our cleanup RELEASE. Native runner launched under HOST-shared
  then UUID-exclusive locks after empty inventory and 8GiB MemAvailable checks.
  Active local SSH handle: 3651. Remote/local artifacts:
  /home/tmc/tmp/nccl-go-testing/native-20260909/ and
  /Users/tmc/tmp/nccl-go-testing/native-20260909/. Poll this same handle; do not
  restart. Four reduced-precision collective suites reached PASS; comparator
  half matrix is running. Runner has a 510-second internal deadline, per-child
  timeouts and process-group cleanup, then final inventory/run.json. After exit,
  retrieve receipts, verify cleanup under fresh locks, and send explicit RELEASE
  to 6E6A46AB-C9A0-4455-B66B-E5D3E4EC97A9. No overall success claimed yet.
- Handle 3651 completed exit 1: half comparator exceeded its 100-second limit.
  Do not restart or poll that handle. Native small-format tests passed 76 modes,
  7,471,104 comparisons, and four omission controls. Other matrices and stream/
  registration tests did not execute. Immediate cleanup inventory retained a
  terminating GPU PID; subsequent fresh both-lock probe verified no compute
  processes or owned groups. run.json/release.json and logs retained locally in
  native-20260909. Cleanup RELEASE sent to 6E6A46AB. Next runner must allow
  sufficient per-matrix time and wait for group/GPU teardown before inventory;
  preserve this timed-out campaign as incomplete. Obtain a fresh slot.
- Prepared half-20260909-02/run.py: complete half matrix max360s, then stream/
  registration tests max65s, 510s overall runner budget, bounded teardown waits.
  Requested a fresh ten-minute slot from 6E6A46AB. It explicitly replied HOLD
  while preparing its own bounded diagnostic and will hand back after cleanup.
  New NCCL run has NOT transferred or launched. Await its explicit handoff;
  previous releases do not authorize this run. DGX.md now records 76 native
  small-format modes separately from the incomplete upstream comparison.
- During HOLD, fixed cleanup to skip a rank's context-bound operations if
  SetCurrent fails, returning an explicit resources-retained error instead of
  operating on the wrong context. Local tests/vet pass. Only run.go differs
  from native-20260909's source manifest. Prepared half runner now enforces
  exact frozen binary hashes before launching; it still targets the previous
  frozen binary, not this cleanup change. HOLD remains; no transfer or launch.
- 6E6A46AB released the requested fresh ten-minute window. Launched prepared
  half-20260909-02 runner under both locks with fresh empty inventory and 8GiB
  MemAvailable check. Frozen prior binaries enforced; newer local graph helpers
  and checkpoint changes are not in this run. Await runner completion, retrieve
  receipts, verify cleanup and send explicit RELEASE to 6E6A46AB before any
  further slot. New graph.go helpers compile locally but are not wired into Run.
- Handle 48108 completed exit 1. Complete half matrix passed 147 cases,
  3,262,224 comparisons, 225 successful child processes; receipt verified.
  Default-stream test passed. Registered-buffer test failed its omission control
  (inspect one-rank in-place identity before attributing to registration).
  Both owned groups empty, runner cleanup_errors empty, fresh both-lock inventory
  empty; cleanup RELEASE delivered to 6E6A46AB. Receipts and logs downloaded to
  half-20260909-02. No jobs queued. Do not poll/restart terminal handle 48108.
- Half, bfloat16, and both FP8 selectors, buffers, scalars, and exact-arithmetic
  oracles are now implemented locally. Local race tests and vet pass. Their
  native collective/omission tests passed on DGX; half also passed its complete
  upstream matrix. BF16 and FP8 complete upstream matrices remain pending.
- The integer-final source manifest describes the previous validated snapshot;
  it no longer matches the current local source. Small-format documentation
  updates and further GPU validation remain pending.
- Corrected registered omission test: one-rank in-place sum is identity, so
  omission detection applies to disjoint poison buffers. Both positive modes
  remain required; local tests/vet pass. Frozen new Linux ARM64 binaries,
  source manifest, and runner in bfloat16-20260909-01. Requested fresh ten-minute
  slot from 6E6A46AB for BF16 matrix max360s then corrected controls max65s.
  No transfer/launch before explicit handoff. Comparator now checkpoints counts
  and validates timing; graph helpers remain unintegrated and unmeasured.
- 6E6A46AB granted BF16 ten-minute window and needs DGX afterward; do not queue
  additional runs without handoff. Launched bfloat16-20260909-01 under both locks,
  fresh empty compute inventory and 8GiB check. Active local handle 79611;
  poll this same handle, never restart because observation yields. Runner checks
  Complete, 78 requested/finished workloads, 147 cases, and 225 successful child
  processes before stream/registration tests. Await terminal run.json, retrieve
  results, verify fresh cleanup, then explicitly RELEASE to 6E6A46AB.
- Handle 79611 completed exit 0. BF16 receipt verified Complete with 78/78
  workloads, 147 cases, 3,262,224 comparisons and 225 successful children.
  Corrected registered-buffer and default-stream tests both passed. Owned
  groups empty and fresh both-lock compute inventory empty; RELEASE delivered
  to 6E6A46AB, which needs DGX next. No remote work queued. Receipts downloaded
  to bfloat16-20260909-01. Next remaining dtype matrices are both FP8 formats;
  require a new coordinated handoff. Never poll/restart terminal handle 79611.
- Prepared local f8e4m3-20260909-01 and f8e5m2-20260909-01 runners/manifests.
  Both enforce the validated BF16 binary hashes, full 78-workload/147-case/
  225-child receipts, max360s matrix and bounded cleanup. No new slot granted,
  no transfer or launch, no queued remote job. DS4 has next host use. Updated
  DGX.md/COVERAGE.md with verified BF16 and corrected registration results.
