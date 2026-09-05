# CUDA allocation-trace prerequisite spike

Registered 2026-09-05T02:43:57.527385+00:00
Purpose: qualify the installed Nsight Systems 2024.5.1 allocation instrument
before constructing a final real-workload null. This is diagnostic; no latency
ratio or null qualification may be inferred from its timings.

One lib 32-output correctness probe under `nsys profile --trace=cuda
--sample=none --cpuctxsw=none --cuda-memory-usage=true --cuda-graph-trace=node`.
Hold the campaign device lock; bound profiler process to120seconds with10second
kill grace. Export SQLite under120seconds. Require nonempty CUDA kernel and
allocation/free activity, native completion, and an allocation schema that
covers application allocations, graph allocations and async pool activity.
An empty/missing trace or unsupported instrument REFUSES this prerequisite.
Do not replace the installed profiler, add a custom runtime callback framework,
or call a driver submission trace physical completion. If the capture succeeds,
inspect schema and scope before devising a per-sample peak algorithm. One repair
may correct an ordinary invocation error before a fresh spike; no substitutions
of instrument or weakened coverage gate. No final null has been sampled.
