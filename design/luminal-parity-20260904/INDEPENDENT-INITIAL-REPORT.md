# Independent initial preregistration review

Verdict: REJECT as authorization to collect accepted performance evidence yet. This is a useful campaign-level budget and scope declaration, not a complete sealed measurement specification. No performance evidence accepted. No production edits, GPU launch, or producer analysis execution occurred. Producer Python was read as data only. This review read the binding brief, adversarial-evidence-falsification SKILL.md, and reference/seal-schema.md.

Reviewed preregistration: `/Users/tmc/tmp/lib-excellence-goal/parity-preregistration-20260904.md`, SHA-256 `6c1e2ac1de49b936491cac99170a8f8dfa1d3c1a2c617925ad7cbe13a05479ba`. References below are to this frozen version. Review consumed less than 10 active minutes of the 20-minute initial budget.

## Canonical historical CUDA failed null

Directory: `/Users/tmc/tmp/lib-excellence-goal/fixed-cuda-20260904T163944/`.

- `protocol.json:13-17`: null median B/A must be within inclusive [0.95,1.05] AND its bootstrap 95% CI must include 1. Both criteria apply to each null.
- `protocol.json:6-12`: 2 warmups; 20 AA, 20 LL, 20 AB pairs; seed 20260904. This historical workload was prefill, not the proposed decode workload.
- `results.json`, LL: median 1.0029391974017794; CI [1.000428068929328,1.0051640285419083]. Magnitude passes; CI fails.
- `decision.json`: NOT_ACCEPTED, assertion names LL; 124 observations.
- `/Users/tmc/tmp/lib-excellence-goal/run_fixed_cuda.py:121-132`: historical implementation uses 5000 median bootstrap draws of 20 paired ratios and selected sorted indices 125,4874. RNG is continued from the seed-20260904 interleaving, not independently reseeded for the CI. The prospective gate is unambiguous; the historical statistical implementation is also available for explicit compatibility decisions before new measurements.
- `/Users/tmc/tmp/lib-excellence-goal/FIXED_CUDA_FINAL_REVIEW.md:18-39` independently records the same failure. This initial audit did not independently recompute historical bootstrap statistics or rehash its entire manifest; values above are historical retained values, not a new sign-off.

Historical artifact hashes:

| File | SHA-256 |
| --- | --- |
| protocol.json | b0906abe221f435da9775feba365f79393011ba17ec6cbc83bf2b524507e80cc |
| results.json | f6cd062a7521c681a5faa330767ad276204ed767ca7a03f5aeb5b1f4d15c4509 |
| decision.json | a045b02ad21fd1e25541f2a0d13fe6d2603e763a05588fc25687506dfd89554c |
| run_fixed_cuda.py | 834fd76f71c469b7841423f833bf0ba61e5f810a625dbace58c93d2cfdb6a3ed |

## Premeasurement defects and required resolutions

1. **Concrete frozen identity is absent** (`preregistration:27`). The sentence says to seal later, but does not itself pin source archive/dependencies/toolchain/executable, device UUID, model-directory manifest plus raw weights/config/tokenizer, or exact build/argv. Require a hashed per-backend execution supplement before collection. Hash the supplement and retain the populated sample IDs and schedule stripped of measured values before first timestamp. No need to pretend these were known at campaign initialization.

2. **Sentinel and semantic liveness gates are absent** (`:25,31`). Merely naming posthoc-sentinel attacks does not freeze cadence, duration threshold, predecessor/self/back-link hashes, temporal brackets and final coverage. Retain original/mutated input and standalone comparator with prospective failure digest. Define whether full output digest covers generated token IDs, every generated logit vector, or both. The exact-equality rule cannot be replaced by an allclose tolerance after failure.

3. **Physical host qualification and serialization are promises, not verifiable gates** (`:31`). Require OS/device lease names and same held lease per arm; numerical load/sentinel/timer limits and freshness window; before/after host checks; stale-evidence refusal probe; named external physical trace and its exact bounded command if required. Application dispatch counts or nvidia-smi snapshots alone are not physical-completion evidence.

4. **Workload and scheduling are underdefined** (`:7,23,25,29`). Prompt length does not freeze exact prompt token IDs. Specify sample count per measured decode/serving cell, pair IDs, bootstrap seed and quantile convention, reset/KV/cache state, prefill versus decode token boundary, EOS policy and sampler. Batch 4 and serving concurrency 4 require explicit request arrival/wave semantics and matched work. Define client location/transport and whether startup/compilation are excluded. Metal's 30 balanced pairs need an explicit schedule algorithm, not only a seed.

5. **CUDA prospective numerical gate can now be explicit** (`:23`). Preserve inclusive [0.95,1.05] AND CI contains 1; it is available. Freeze the new real-workload schedule and statistic implementation before measuring. The generic 10000 bootstrap sentence (`:29`) differs from the historical implementation; make its intended scope explicit so historical gate recovery cannot conceal statistical substitution. Do not transplant the historical prefill control as decode qualification. The 45-minute diagnosis and two repairs must not become two failed final nulls followed by a selected passing third attempt. One final null only; failure closes.

6. **Calibration precedes an unstated control** (`:27`). The requested upstream/instrumented 10% calibration itself compares timings. State explicitly that corresponding real-workload null qualification precedes its ratio, and freeze its sample count, timer and gate statistic. Tuning may use search timings but cannot contribute an accepted parity ratio. Retain candidate count, search seed/settings, selected plan, and evidence of actual exploration beyond one plan.

7. **Peak memory parity is underspecified** (`:25`). Freeze raw units, process/device allocation scope, peak-reset policy, aggregation statistic and relative-difference denominator. Runtime-resident versus per-request incremental memory are different quantities. The fixed <=5% rule remains binding once measured and must not be changed to rescue divergent runtime overhead.

8. **Seal closure is missing** (`:27,31`). Freeze retention of binaries, source/dependency artifacts, all model files/manifests, mutation/comparator artifacts, per-interval files, schedule, chain, build and run transcripts, before/after identity hashes, and whole archive hash. Require archive-only verification after fresh extraction on a different machine and separate CUDA/Metal audit verdicts. A bare list of hashes does not reconstruct an omitted artifact.

9. **Noise-floor/decision linkage needs a fixed rule** (`:23,29`). Metal allows CV up to 5% yet declares a beat at CI lower bound >1.02. Skill requires A/A floor reporting and every claimed effect to exceed that floor; define its computation and effect test before collection rather than interpret it after seeing the outcome. A null interval containing one is not itself a numerical floor.

## Scope judgment

No objection to closing cells with a concrete missing prerequisite within their frozen budget, or recording negative qualified evidence. The selected F32 8-prompt/32-generated B1/B4 scope, both backend separation, no projection-fusion reentry, no deletion, and no push match the binding brief. This review does not require implementing unsupported workloads beyond the stated construction budgets.

Do not enter measurement until a per-backend supplement resolves the applicable items. If feasibility already proves the required runtime/shape absent and the construction budget closes, record REFUSE and the exact missing prerequisite; no seal or benchmark needs to be fabricated for a closed axis.
