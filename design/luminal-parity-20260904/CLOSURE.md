# Luminal parity campaign: terminal dispositions

Recorded 2026-09-05T02:47:20.670863+00:00

This campaign ends with no accepted tuned decode or serving performance claim.
It does not prove that llama-infer is slower, faster, equal, or intrinsically
unable to beat Luminal. It exercises the user's permitted terminal outcome:
CLOSED / REFUSE where the registered comparison cannot be constructed with the
available instrument and host. No closed row is a deferred automatic retry.

## Scope and disposition

All registered cells specify the Llama-3.2-1B-Instruct F32 checkpoint, eight explicit
prompt tokens and 32 generated tokens. B4 means independent requests, not four
positions of one sequence. Serving C1/C4 means client submission through final
token. CUDA and Metal remain separate.

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

## Evidence that determines the CUDA refusal

The premeasurement plan is `parity-preregistration-20260904.md`, with original
hash file and commit39ade53167a4e89a79a429fc675ee8e5235b5b4a. The trace spike was
registered before invocation in `parity-20260904-memory-spike.md`. It required
nonempty CUDA kernels and allocation/free activity and explicitly refused an
empty or unsupported instrument, without installing a replacement profiler or
building a custom callback framework. Nsight2024.5.1 reported:

- Installed CUDA driver version13.0 is unsupported by this build.
- It loaded its CUPTI12.6 library.
- No CUDA events collected.

The instrumented application nevertheless exited0 and completed its32 outputs.
The exported SQLite contains diagnostics and metadata/enumerations but NO CUDA
activity tables. This is an infrastructure/instrument refusal, not a candidate
performance loss. A zero trace is not a zero-operation GPU execution: the
application generated outputs, but physical tracing failed.

The separate default-pool counter alternative was inspected and rejected as
incomplete: Luminal runtime.rs654-668 handles a distinct CUDA graph allocation
domain and runtime.rs820-835 queries an allocation's actual pool. No sum of
independent high-water marks was relabeled a simultaneous runtime peak.

This missing trace prerequisite closes the new null BEFORE sampling it.
The historical failed Luminal/Luminal null remains closed unchanged. Its
[0.95,1.05] median gate AND interval-containing1 rule were recovered and never
loosened. No new comparative ratio is reported. The new null was not retried
until favorable; it did not run because its evidence prerequisite failed.

## Tuned construction result, not performance evidence

Both retained probe executables completed. Independent checking found exactly
4,104,192 finite values per arm, all32 argmax tokens equal and zero failures
at atol=rtol1e-4. Worst absolute difference1.9073486328125e-05. Full logit bytes
are NOT identical; only the generated token sequence is exact. The retained
+1 logit mutation tests the offline comparator's sensitivity, not live model
consumption. Search logs prove exploration beyond one plan and100 iterations
per bucket; they do not prove200 distinct valid graphs. See
`parity-20260904-falsifier/CONSTRUCTION-REPORT.md` for exact boundaries.

No deployment speed, decode tokens/sec, serving latency, match, or loss is
inferred from these construction logs or their test durations.

## Budgets and STOP rules

Budgets are upper bounds, not instructions to spend the full allocation after
a decisive prerequisite failure. No phase claims budget exhaustion. These
conservative wall envelopes include unrelated work and therefore upper-bound
active effort. No second implementation or optimization candidate was opened.

| Phase | Registered maximum | Observed conservative envelope | Disposition |
| --- | --- | --- | --- |
| Inventory and coordination |45min |02:27–02:47Z, <=20min |Source and host prerequisites identified |
| CUDA tuned construction |60min,2repairs |02:32:33–02:42Z, <10min |Build and32-output construction pass,0repair cycles |
| CUDA instrumentation |60min,2repairs |02:42–02:47Z, <=5min |Default pool incomplete; one Nsight spike categorically unsupported; no invocation repair applicable |
| CUDA null diagnosis/preparation |45min,2repairs |02:27–02:47Z, <=20min inclusive upper bound |Prior pairs already balanced10AB/10BA; no spurious order-count fix; final null not entered |
| Metal host availability |30min external-wait maximum |02:30–02:44:14Z, <15min |Two route failures trigger two-failure host STOP before deadline |
| Metal construction/null/measurement |60/30/60min |0GPU minutes |Closed on host prerequisite |
| Qualified comparison per backend |60min each |0samples |Never entered without passing null |
| Independent/final audit |45min |Initial <10min; source <15min; construction <10min; closure <=10min |Total maximum45min; reports retained |

Metal's two-failure stop follows the loaded adversarial-evidence-falsification
skill (`/Users/tmc/.claude/skills/adversarial-evidence-falsification/SKILL.md`):
“Host qualification fails twice — record UNMEASURED and stop; do not discount
the gate.” The application here treats a route that cannot reach the designated
host as failed host qualification. The terminal campaign status is CLOSED /
REFUSE, consistent with the binding brief, not a promise to retry later.

## Retention and ownership

Diagnostic probe archive SHA256:
8f656f4caff1f42425dd73e4ba09e8849271e5d2f8720136bfb58646c4faf9d2.
Diagnostic trace archive SHA256:
301bf3d225d054e95f789631873f40d80ad09bc11dfef97eda4434519768da23.
Both whole archives and their listed files were verified after local transfer.
These are diagnostic archives, not complete accepted performance seals.

Isolated branches: codex/luminal-parity-20260904 and codex/parity-cuda-20260904.
No production optimization, unused-code deletion, projection-fusion work,
changes to protected worktrees, or push. The existing main go.work.sum and
.claude/ dirt remain. Diagnostic harness source is retained in the raw artifact directory and
parity-20260904-lib-probe; it is not promoted into production.

All campaign GPU processes exited. A final nvidia-smi compute-app query was
empty, and release was sent to session54B33A62. No process/lock is treated as a
pending task. Documentation commits use the mandated helper fallback after
two missing-key failures, with git notes recording available provenance.

Independent closure audit: CUDA SIGN-OFF ON REFUSAL; Metal SIGN-OFF ON
REFUSAL. See `parity-20260904-falsifier/CLOSURE-REPORT.md`. No performance
claim is accepted by either verdict.

The completion audit also found overbroad E-Graph megakernel labels. Commit
fb4df36f596db3acb3c86c0e64e52e7f343676cc clarifies CLI help and Metal logging;
f16 remains explicitly described as discrete kernels. Only strings changed,
not inference behavior or dispatches. Linux ARM64 build passed; gofmt parsing
and git diff --check passed for both changed source files. Changes are on the
isolated lane, not main. The construction probes precede this text-only commit
and are not claimed as a new-head physical measurement.

## Additional GPU ownership check

The user identified session2379 as another possible GPU owner during closeout.
Its screen shows CUDA qualification work. A wrapped request was sent asking
which host it used and whether work overlapped02:34–02:44Z. No request to stop
its work was sent. Our probes held their own cooperative lock; that does not
establish exclusivity against sessions using a different lock. Before-launch
empty compute-app snapshots establish only those instants. No null or
performance evidence was accepted, so even confirmed contention cannot turn
these diagnostic archives into an accepted timing result. No further GPU work
is planned; any future campaign must coordinate with both known owners.

Session2379 replied with attempt05 at02:41:08.973–02:41:14.750Z and an
NVRTC-failed attempt04 at02:36:54.065–02:36:54.989Z. Original tar member
timestamps place our Luminal exit-code at02:36:18Z and lib exit-code at02:37:56Z;
our Nsight capture metadata starts02:43:58Z with exit-code at02:44:05Z.
The reported05 is within the broad coordination window but between our
recorded executions, so direct GPU overlap is not established. The lock still
provides only cooperative exclusion. Both peer and coordinator received these
narrowed times;2379 received54B33A62's full session ID for future ownership.
No current run needs to wait on this closed campaign.

The temporary lib test was moved out of the package into the retained probe
source directory after execution: it intentionally requires a model environment
and should not enter the normal test suite. No source was discarded.
