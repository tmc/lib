# Independent final closure audit

CUDA: **SIGN-OFF ON REFUSAL**.

Metal: **SIGN-OFF ON REFUSAL**.

These verdicts accept the terminal disposition, not any performance evidence. No measured speed loss, match, or superiority is established. This independent read-only audit used newly written `check_closure.py` and direct SQLite queries; it did not execute producer inspection/analysis code or any GPU workload. Completed within the 10-minute closure allowance.

Reviewed closure: `/Users/tmc/tmp/lib-excellence-goal/parity-20260904-CLOSURE.md` (recorded 2026-09-05T02:47:20.670863+00:00). Supporting independent reports: INITIAL-REPORT.md, UPSTREAM-FEASIBILITY.md, CONSTRUCTION-REPORT.md. Machine-readable archive and database audit: `closure-check.json` beside this report.

## Direct verification

Both whole diagnostic archive hashes match closure lines 101-104:

- cuda-probe archive: `8f656f4caff1f42425dd73e4ba09e8849271e5d2f8720136bfb58646c4faf9d2`.
- memory-spike archive: `301bf3d225d054e95f789631873f40d80ad09bc11dfef97eda4434519768da23`.

I independently parsed each SHA256SUMS from inside the archive, rehashed every listed file from archive bytes, and checked the corresponding locally transferred file. All 6 probe entries and 8 trace entries match. Probe entries include both executables, both full logit streams and both run logs. This checks integrity of the listed diagnostic artifacts, not a complete provenance seal or omitted exit-code files in the probe archive. Raw full-logit hashes and independent numerical results remain as previously reported in CONSTRUCTION-REPORT.md.

I opened `parity-20260904-memory-spike/capture.sqlite` read-only and enumerated every table and its row count. There are no CUPTI_ACTIVITY_KIND tables and no CUDA kernel, memory allocation/free or other CUDA activity tables under another name. CUDA enum tables merely define values. DIAGNOSTIC_EVENT contains 10 rows, including:

- Installed CUDA driver version 13.0 is not supported by this Nsight Systems build.
- Loaded CUPTI path ends in `libcupti-sbsa.so.12.6`.
- No CUDA events collected.

Database capture metadata records host spark-31db, CUDA tracing true, GPU memory collection true, CUDA graph mode Node, and the lib correctness-probe argv. Session start is 2026-09-05T02:43:58Z, after the spike registration's stated 02:43:57.527385Z. That order is internally consistent; local timestamps alone do not cryptographically prove prospective publication. The trace run log has all 32 output rows and PASS; its exit-code file is zero. Empty activity is thus a failed tracing prerequisite, not evidence of no GPU work.

The registered spike explicitly requires nonempty kernels/allocation activity and refuses unsupported/empty capture, forbids profiler replacement or a custom callback framework, and permits only an ordinary invocation repair. The observed unsupported-driver diagnostic and absent CUDA events satisfy the written STOP. Default-pool counters are not an adequate replacement, for the distinct graph allocation domain identified in CONSTRUCTION-REPORT.md.

Metal route evidence consists of the initial `parity-20260904-hosts.txt` error and the second JSON attempt at 02:44:13Z with exit code 1, empty stdout, server-hello closure and no route to host. The initial text lacks its own structured timestamp/argv, so it is weaker provenance than the second attempt; it does not establish a successful connection. Neither file supplies physical Metal evidence, and the closure correctly credits zero Metal samples. The two-failure host stop is explicitly attributed to the loaded skill and its application to failed routes.

## Challenge: early stop and exact-output scope

The broad phase budgets are maxima, not minimum work quotas. Closure lines 76-97 do not claim exhaustion: CUDA is stopped by the prospectively registered unsupported-instrument rule; Metal is stopped by the skill's two-failure host qualification rule. Remaining minutes do not repair an explicitly excluded profiler upgrade, and the missing null/evidence prerequisites make B4/serving construction insufficient for an accepted comparison. Early closure is therefore consistent with these rules. It must not be described as proof that no other instrument, host, or implementation could ever succeed.

The phrase “all cells use” describes the registered scope; it does not mean unrun cells executed. Closure's table explicitly marks unrun cells and zero null/measurement samples. CUDA tuned construction passed a 32-token semantic diagnostic only. Closure lines 62-68 explicitly distinguish exact generated tokens from differing full logit bytes and identify the mutation as offline corruption sensitivity. That prevents the looser diagnostic allclose from silently becoming the campaign's exact-digest gate. No exact-full-logit performance acceptance is present.

## Per-backend attack disposition

| Attack | CUDA | Metal |
| --- | --- | --- |
| Stale binary / source identity | Archive binaries rehash, but full source/build/toolchain binding is not certified. Missing performance seal remains REFUSE. | No current device binary execution claimed; REFUSE. |
| Stale model | Full reconstruction/model-consumption seal absent. Numeric construction agreement is not a model identity seal; no performance acceptance. | No fresh model execution claimed; REFUSE. |
| Posthoc sentinel | No qualified campaign sentinel chain exists. Closure does not present the diagnostic logs as one; REFUSE. | No campaign or sentinel chain; REFUSE. |
| Arm-label swap | Independent retained vectors/log rows agree, but no physical asymmetric arm identity is certified. Cannot accept performance. | No measured arms; REFUSE. |
| Reduced work | Independent raw check confirms 32 complete vectors/arm and 32 matching argmax tokens; this does not establish B4 or serving. | No work credited; no cross-backend substitution. |
| Cache effects | No qualified reset/cache schedule evidence is credited; no timing comparison accepted. | No samples. |
| Async timing / physical-work substitution | SQLite directly proves instrument failed to collect CUDA activity. Application completion cannot replace it. Refusal is correct. | No physical timing/completion evidence; refusal is correct. |
| Duplicate samples | Construction rows have exact indices 0-31. They are not promoted to randomized repeated measurements; qualified sample count is zero. | Zero samples. |
| Threshold substitution | Historical null [0.95,1.05] plus CI contains 1 retained; new null never sampled. Unsupported spike not rescued by weaker counter. Full-logit inequality disclosed. | Failed host qualification not replaced with historical Metal evidence or CUDA evidence. |

## Terminal coverage and remaining integration audit

The closure table covers tuned comparator B1/B4, real-workload null for decode B1/B4 and serving C1/C4, decode throughput B1/B4 and serving C1/C4 separately for both CUDA and Metal. Each cell has an explicit CLOSED / REFUSE disposition with a missing prerequisite. CUDA B1 construction success is retained without claiming an accepted comparator. It asserts no speed ordering and does not repeatedly re-enter the failed historical null.

This signs off on the refusal narrative and diagnostic integrity only. I did not independently verify the final Git notes, remote GPU release, clean-process query, protected-worktree preservation or final PERFORMANCE_LEDGER integration in this bounded evidence audit. The coordinator must inspect those authoritative artifacts before goal completion. Their omission from this sub-audit is not performance acceptance or evidence that those actions did not occur.
