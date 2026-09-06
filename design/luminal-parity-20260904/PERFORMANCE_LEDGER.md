# Campaign ledger extract

Canonical ledger: ~/tmp/lib-excellence-goal/PERFORMANCE_LEDGER.md

## Luminal parity campaign 2026-09-04: construction evidence

Preregistered before execution in `parity-preregistration-20260904.md`,
committed as `39ade53167a4e89a79a429fc675ee8e5235b5b4a` in the isolated
`codex/luminal-parity-20260904` lane. This entry accepts no performance ratio.

CUDA B1 construction probe completed on spark-31db under a cooperative device
lock and a 1200-second deadline. The newly built Luminal adapter used search
seed 20260905, limit 100, trials 3, keep-best 4 and separate prefill/decode
buckets. The retained log reaches 100/100 for both buckets; this records search
iterations, not a claim that every candidate was distinct. Both Luminal and
current lib generated the same complete 32-token sequence from the registered
eight-token prompt. All 4,104,192 finite logits satisfy the prospective
atol=rtol=1e-4 diagnostic contract; worst absolute difference 1.90734863e-05.
The retained +1 first-logit mutation is rejected. Independent construction
review PASS is recorded in `parity-20260904-falsifier/CONSTRUCTION-REPORT.md` (semantic scope only). These are semantic probe outputs, not timed samples.

Remote probe archive SHA-256:
`8f656f4caff1f42425dd73e4ba09e8849271e5d2f8720136bfb58646c4faf9d2`.
Archive verified locally; evidence is `parity-20260904-cuda-probe/`, source
archive is `parity-20260904-luminal-source.tar.gz`. This diagnostic archive
is not a complete performance seal.

The CUDA A/A, calibrated B1 decode, B4 and serving axes remain unfinished.
`parity-20260904-falsifier/UPSTREAM-FEASIBILITY.md` identifies the exact
unmodified-upstream calibration prerequisite: shipped entrypoints cannot
select this workload, and neither supplies a client-serving lifecycle.
B4 requires independent request histories, not sequence dimension four.
Metal has no new probe: its registered remote route returned No route to host.
The construction result does not qualify that backend.

## Luminal parity 2026-09-04: terminal axis dispositions

All registered cells are now CLOSED / REFUSE under the concrete prerequisite
failures below. This is no accepted performance claim and no proof of a speed
loss. A new CUDA same-runtime null was not sampled: its complete peak-memory
instrument prerequisite failed. The old failed null is unchanged. Zero new
comparison ratios were admitted; no gate was loosened or resampled to pass.

| Backend | Axis / cells | Terminal disposition | Specific prerequisite and evidence |
| --- | --- | --- | --- |
| CUDA | Tuned comparator, B1 | CLOSED / REFUSE as accepted performance comparator; construction succeeded | 100 search iterations in each of two buckets with four finalists each; complete 32-token/full-logit semantic check independently passes. Missing complete memory instrument and exact unmodified-upstream calibration. CONSTRUCTION-REPORT.md and UPSTREAM-FEASIBILITY.md |
| CUDA | Tuned comparator, B4 | CLOSED / REFUSE | Existing graph has one causal history and no independent request mask/cache boundary; true B4 needs construction. Accepted comparison also lacks memory instrument and upstream calibration. UPSTREAM-FEASIBILITY.md |
| CUDA | Real-workload null, B1/B4 decode and C1/C4 serving | CLOSED / REFUSE, no new qualified null sampled | Default-pool peak excludes graph allocation domain. Registered Nsight memory prerequisite produced zero CUDA activity, with explicit CUDA13 unsupported diagnostic. No complete per-sample peak-memory source; B4/serving additionally lack matched workload adapters. memory-spike/diagnostics.json and inspection.json; CONSTRUCTION-REPORT.md |
| CUDA | Decode throughput, B1/B4 | CLOSED / REFUSE, no comparison ratio | Required passing null and complete matched peak-memory evidence unavailable; exact unmodified-upstream application cannot select F32 1B/8/32. Source audit plus allocation-trace refusal |
| CUDA | End-to-end serving, C1/C4 | CLOSED / REFUSE, no comparison ratio | Luminal entrypoint processes one local prompt and exits, without client lifecycle/queue/arrival/final-token boundary; prerequisite null, memory and calibration unavailable. UPSTREAM-FEASIBILITY.md |
| Metal | Tuned comparator, B1/B4 | CLOSED / REFUSE | Registered mote @mac2 host unreachable on both attempts; no fresh physical tuned comparator. parity-20260904-hosts.txt and parity-20260904-metal-route-second.json |
| Metal | Real-workload null, B1/B4 decode and C1/C4 serving | CLOSED / REFUSE, zero samples | Same unavailable host; no passing current-backend A/A. Historical Metal prefill or CUDA output cannot satisfy it |
| Metal | Decode throughput, B1/B4 | CLOSED / REFUSE, no comparison ratio | No reachable registered GPU host or current null; unmodified exact-workload calibration also absent. Route evidence and UPSTREAM-FEASIBILITY.md |
| Metal | End-to-end serving, C1/C4 | CLOSED / REFUSE, no comparison ratio | No reachable registered host, matched serving adapter, or current null. Source audit identifies one local response loop, not a service |


The installed Nsight2024.5.1 trace exited0 but its diagnostic explicitly rejects
driver13.0 support and reports no CUDA events. Exported SQLite has no CUDA
activity tables. Trace archive SHA256:
`301bf3d225d054e95f789631873f40d80ad09bc11dfef97eda4434519768da23`.
Two Metal route probes failed before the30-minute deadline, invoking the
registered skill's two-failure host STOP. All phases stayed within upper
budgets; none claims exhaustion where the prerequisite caused an earlier STOP.

Full reasoning, per-phase budget accounting, preserved construction evidence,
source citations and ownership audit: `parity-20260904-CLOSURE.md`.
Independent closure review: CUDA SIGN-OFF ON REFUSAL and Metal SIGN-OFF ON
REFUSAL, in `parity-20260904-falsifier/CLOSURE-REPORT.md`. No accepted
performance seal exists. All rows above are final, not deferred.

Text-only follow-through: isolated commit `fb4df36` replaces overbroad
megakernel help/log labels with E-Graph operation terminology and explicitly
states the f16 discrete-kernel behavior. No inference or performance change.

Additional ownership record: session2379's reported02:41:08–02:41:14Z GPU
qualification falls between this campaign's recorded probe executions. Broad
campaign-window overlap is not proof of concurrent kernels. Cooperative-lock
limits and exact archive timestamps are retained in the closure report.
