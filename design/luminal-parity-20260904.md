# Luminal decode and serving parity preregistration

Registered: 2026-09-05T02:27:27.404244+00:00
Binding source: LUMINAL-PARITY-BRIEF.md. No benchmarks run in this continuation before this registration.

## Objective and matrix
Report CUDA on spark-31db and Metal on tm4sm.local separately. For each backend resolve tuned Luminal availability, decode throughput, end-to-end serving, and real-workload A/A null qualification. Search-limit-1 is ineligible. Primary requested cells: Llama-3.2-1B-Instruct, matching checkpoint tensors and tokenizer, F32, prompt length 8, generated length 32, batch/concurrency 1 and 4. Source feasibility must confirm these shapes before execution; unsupported cells close with prerequisites, never silently substitute prefill. No cross-backend inference.

## Budgets and termination
Wall time is active work per phase; hardware reservations and external waits have a separate 30-minute deadline. Record start/end and consumed budget. Existing evidence may establish feasibility but cannot replace a fresh sealed matched run.

| Phase | Budget | Terminal rule |
| --- | --- | --- |
| Inventory, source feasibility, skill lookup and host coordination | 45 minutes | Name absent prerequisites per backend/cell |
| Tuned comparator construction per backend | 60 minutes, at most 2 repair cycles | Freeze tuned plan or close comparator and dependent cells |
| Timer, correctness, checkpoint and memory instrumentation per backend | 60 minutes, at most 2 repair cycles | Complete seal and parity gate or close dependent cells |
| CUDA null diagnosis and repair before any comparison | 45 minutes, at most 2 repair cycles before final null | One final preregistered null; failure closes CUDA arm |
| Metal null preparation | 30 minutes | One final preregistered null; failure closes Metal arm |
| Measurement per qualified backend | 60 minutes | Record every sign; timeout is incomplete evidence, no ratio |
| Independent falsification and final ledger audit | 45 minutes | Accept only independently surviving evidence; otherwise close with defect |

## Gates fixed before measurements
Recover the prior CUDA null numerical gate and preserve it exactly; if unavailable, close CUDA qualification rather than invent a replacement. For Metal A/A: paired median B/A in [0.98,1.02], bootstrap 95% interval contains 1, per-arm CV <=5%. Use 30 balanced randomized AB/BA pairs, fixed seed 20260904, 5 untimed warmups per arm, identical reset/cache policy. CUDA uses prior gate plus fixed schedule/cardinality documented before execution; never resample to obtain a pass.

Every sample must contain complete output digest, output count, peak memory, workload identity and synchronized elapsed time. Match full output digests and peak-memory accounting definition; peak-memory relative difference <=5%, with raw bytes retained. Exact digest mismatch closes the comparison. No truncation, asynchronous timing, missing samples, duplicate samples or arm swaps accepted. End-to-end serving timing includes client request through final token and records TTFT, TPOT, aggregate tokens/s and p50/p95 latency. Decode excludes prefill equally and synchronizes GPU completion. No ratios before corresponding real-workload null passes.

Seal source revision and dirty patch, dependency locks, compiler and driver versions, binaries, tuned plan/search configuration, checkpoint/tokenizer hashes, workload/schedule and analysis code before execution. Tuning requires search budget >1, default 100 explored candidates or 20-minute search cap, retained search log and selected plan; no claim of globally optimal tuning. Calibrate instrumented comparator against unmodified upstream within 10% under matching timer scope or close cross-implementation claims.

Ratios use Luminal elapsed/lib elapsed, paired median and seeded 10000-replicate bootstrap 95% CI. Match: CI inside [0.98,1.02]; beat: lower bound >1.02; slower: upper bound <0.98; otherwise inconclusive. Record all outcomes. No production optimization before a qualified profile identifies a bottleneck; projection fusion excluded. This campaign cannot detect defects present in the baseline without its external reference. All unrelated benchmarks excluded until this campaign closes because no production change is yet proposed. Any proposed production change requires coverage inventory and protected matrix preregistration before execution.

Independent falsifier uses separate analysis and attacks stale binary/model, posthoc sentinel, label swap, reduced work, cache effects, async timing, duplicate sample and threshold substitution. Preserve all evidence and failed attempts. Do not delete unused code, touch protected worktrees, overlap GPU work, or push. Commit scoped documentation in an isolated lane using the authorized commit tool; add git notes. PERFORMANCE_LEDGER.md receives explicit terminal rows for every backend and axis, including unresolvable prerequisites.
