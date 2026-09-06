
## Luminal parity campaign 2026-09-04: construction evidence

Preregistered before execution in `parity-preregistration-20260904.md`,
committed as `39ade53167a4e89a79a429fc675ee8e5235b5b4a` in the isolated
`codex/luminal-parity-20260904` lane. This entry accepts no performance ratio.

CUDA B1 construction probe completed on spark-31db under an exclusive device
lock and a 1200-second deadline. The newly built Luminal adapter used search
seed 20260905, limit 100, trials 3, keep-best 4 and separate prefill/decode
buckets. The retained log reaches 100/100 for both buckets; this records search
iterations, not a claim that every candidate was distinct. Both Luminal and
current lib generated the same complete 32-token sequence from the registered
eight-token prompt. All 4,104,192 finite logits satisfy the prospective
atol=rtol=1e-4 diagnostic contract; worst absolute difference 1.90734863e-05.
The retained +1 first-logit mutation is rejected. Independent construction
review is pending. These are semantic probe outputs, not timed samples.

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
