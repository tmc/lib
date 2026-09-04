# Verifier-guided Metal kernel experiment

This document describes a bounded experiment in which an agent edits one Metal
kernel, an independent harness checks its output, and repeated device runs rank
the admitted candidates.

Status: experimental design. It is not part of the `llama-infer` execution
architecture and does not establish an end-to-end inference improvement.

## Scope

Start with RMSNorm over a fixed family of shapes:

```text
y[row, col] = x[row, col]
              * rsqrt(mean(square(x[row, :])) + epsilon)
              * weight[col]
```

The candidate may change Metal source and declared launch geometry. It may not
change inputs, the reference implementation, tolerances, sampling policy,
measurement code, or accepted shape set.

RMSNorm is suitable because the semantic reference is small and independent.
A deliberately scalar or redundant baseline can prove that the optimization
loop detects an obvious improvement. Performance gains over that baseline do
not establish that the result beats the production RMSNorm implementation.

## Harness boundary

```text
frozen corpus and reference
          |
          v
candidate source -> compile -> execute -> compare -> benchmark
          ^                                      |
          +-------------- feedback --------------+
```

The harness owns:

- fixed-seed inputs covering aligned, tail, small, and large dimensions;
- an independent numeric reference;
- dtype- and shape-specific tolerances;
- timeout and resource limits;
- warmup and interleaved baseline scheduling;
- raw result retention; and
- final promotion tests on inputs withheld from the optimization loop.

The reference is an independent numerical oracle, not an “exact” oracle for
floating-point reduction order. It uses an absolute and relative error policy
chosen before the search starts.

Keeping files outside the candidate working directory prevents ordinary edits,
but it is not a security boundary when the agent process has the same filesystem
permissions. Adversarial evaluation requires process or OS-level isolation,
read-only mounts, and a restricted tool surface.

## Candidate result

Each attempt records:

- source hash and launch geometry;
- compiler and device identity;
- compile, launch, evaluation, timeout, or correctness outcome;
- maximum absolute and relative error by shape;
- warmup and individual timing samples; and
- baseline samples interleaved with the candidate.

Compilation may be lazy in the selected Metal/MLX API. The executor must force
evaluation and report that error rather than treating source construction as a
successful compile. Before implementation, verify the current `mlx-go` API and
error behavior against the pinned dependency revision.

## Measurement

Synchronize the device before starting a sample and before stopping it, or use a
supported device timestamp interval with the same boundaries. Report the full
sample distribution and candidate-to-interleaved-baseline ratio.

Use `ns/op` or `kernel/s` for one kernel invocation. The formula
`1e6 / median_microseconds` produces kernel invocations per second, not
mathematical operations per second. Report FLOPs/s only after defining and
versioning the exact operation count.

Separate compilation time, first-run time, and steady-state time. Thermal state,
power mode, competing GPU work, and device identity belong in the retained run
metadata.

## Correctness and anti-Goodhart controls

- Compare every candidate before timing it.
- Reject NaN, infinity, and untouched-output sentinels explicitly.
- Include a mutation control that proves the output depends on each required
  input.
- Include awkward dimensions that exercise vector tails.
- Re-run finalists on withheld seeds and shapes.
- Keep the accepted tolerance and corpus version in the result.
- Reject candidates that inspect the corpus or special-case known values.
- Re-run the production reference and the original baseline in the same process.

A compiler failure, timeout, empty result, skipped GPU run, or missing device is
not a slow candidate; it is a distinct non-admitted outcome.

## Promotion boundary

A candidate may be promoted from the experiment only after it:

1. passes the full frozen and withheld correctness suites;
2. handles every declared layout, alignment, and tail case;
3. reports its device resource use and supported chip range;
4. improves a production baseline, not only the deliberately naive seed;
5. improves the end-to-end plan on a real model and workload; and
6. retains a fallback for unsupported shapes or devices.

Fused MoE and other multi-operation kernels require separate semantic and
resource designs. Success on RMSNorm demonstrates the harness, not the
generality of any generated optimization.
