# Independent CUDA construction correctness report

Verdict: PASS for the narrow retained-vector semantic construction check. NO performance acceptance, null qualification, memory-parity acceptance, serving or B4 result. No GPU work or production changes. The independent checker `check_construction.py` was authored here from the contract and raw artifacts; no producer analysis program was executed or imported. Audit completed within 10 active minutes.

Input directory: `/Users/tmc/tmp/lib-excellence-goal/parity-20260904-cuda-probe/`.
Contract: `/Users/tmc/tmp/lib-excellence-goal/parity-20260904-correctness-contract.md`.
Independent machine-readable result: `construction-check.json` beside this report.

## Numerical construction result

Each stream has exactly 16,416,768 bytes: 32 complete vectors of 128,256 little-endian F32 values, totaling 4,104,192 values per arm. All values in both streams are finite. Every paired element satisfies `abs(luminal-lib) <= 1e-4 + 1e-4*abs(lib)`. There are zero failing positions; maximum absolute difference is 0.000019073486328125.

All 32 independently recomputed greedy argmax tokens match across arms. The full indexed token/cardinality rows from both logs also match the recomputed tokens exactly, with indices 0 through 31 and no extras. Both exit-code artifacts contain zero. The exact sequence is retained in `construction-check.json`.

The raw logit bytes are NOT equal. Luminal SHA-256 is `9f3a32ce9fe6f78aa84298705b7ca3ebcff0659471ca00b330a0ba1797ba1e4f`; lib SHA-256 is `f5e12bc867575d3d8bfa11664172ac0e3937f816da7619ef37700e4240c5ea7f`. Therefore this does not pass any exact-full-logit-digest gate. It does prove exact equality of the generated 32-token sequence. Do not conflate those output definitions.

Independent mutation: add exactly +1 to the first Luminal logit, round to F32, retain the complete changed stream as `mutated-first-logit.f32`, then run the SAME numerical comparison function. It fails exactly at flattened position 0 (step 0, vocabulary index 0), with maximum absolute difference 1.0. Mutation artifact SHA-256: `5d17aebf745aad911f8b22f77eca880837c00881ddbeda84ef14beb617ad3c1f`. This demonstrates comparator sensitivity to retained-output corruption, not a live on-device weight mutation or proof of model consumption.

## Search exploration evidence

Raw `run.log:54659` declares seed 20260905, limit 100, trials 3, keep_best 4. Progress reaches 100/100 at `:78782` and `:86171`, for the two search buckets. Both buckets retain four deployment finalists (`:78786-78789`, `:86175-86178`), with distinct recorded finalist measurements; logs also record faster/slower search progress and candidate rejections. `:86183` records search completion, followed by all 32 output steps at `:86184-86215`.

This establishes actual exploration beyond search-limit-1 and completed search iteration limit 100 per bucket. It does NOT establish 100 unique valid candidate graphs per bucket: rejected candidates and revisits are possible and the inspected lines provide no unique graph-identity count. Report “100 search iterations per each of two buckets, multiple finalists,” not “200 unique candidates.” No search timing is promoted to a comparison ratio.

## Default-pool high-water mark is not a complete peak-memory proof

Source root: `/Users/tmc/tmp/lib-excellence-goal/parity-20260904-luminal/crates/luminal_cuda_lite/src/`.

`runtime.rs:820-835` allocates the shared arena and queries the allocation's actual `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`. `:850-864` trims the CURRENT pool, which need not be the default pool. Thus a default-pool-only query first needs evidence that all relevant allocations actually used that pool.

There is a second allocation domain explicitly handled in this build: `runtime.rs:654-668` documents the separate graph allocator and invokes `cuDeviceGraphMemTrim`. `kernel/cuda_graph.rs:352-356` instantiates CUDA graph executables through `cuGraphInstantiateWithFlags`. The raw selected-graph summary (`run.log:86217`) demonstrates this execution architecture is used. Therefore even correct accounting of the stream-ordered pool does not by itself account for graph-owned memory.

Additional runtime-owned resources include loaded kernel modules (`kernel/hlir.rs:259`, `kernel/generic_matmul.rs:397`) and the cuBLASLt handle cache (`lib.rs:415-435`). Their device allocation contribution is not quantified by this static review. I found no evidence that the default pool's USED_MEM_HIGH fully includes these resources. A narrower metric may legitimately be called “default stream-ordered pool used-memory high-water mark,” but cannot be relabeled complete model/runtime device peak without an allocation-domain audit and matched accounting on lib.

Do not sum independent domain high-water marks and call the result an observed simultaneous peak: peaks can occur at different times. Full peak-memory parity remains unqualified with the current artifacts. The exact unmodified-upstream calibration, real-workload null, output-digest interpretation and all seal gates remain separate prerequisites.
