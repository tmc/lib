# Reference coverage audit

Reference: nccl-tests `b4d5beebca8a76cf01335f724d154b9b9d394d96`.
The inventory below was checked against `src/common.cu:1198` (option table),
`src/common.cu:1421` (help), and `src/alltoallv.cu:153` (traffic controls).
This is a work list, not a declaration that missing features are out of scope.
Physical results and receipt locations are in [DGX.md](DGX.md).

| Reference control | Go implementation and remaining evidence |
| --- | --- |
| All eleven host operation executables | Implemented; complete single-GPU comparisons retained for ten datatypes; native collective tests cover all twelve. Multi-GPU execution unmeasured. |
| `-t`, `-g`, MPI | One OS thread, one rank per distinct local device. Multiple threads and MPI not implemented. Only one physical rank measured. |
| `-b`, `-e`, `-i`, `-f` | Individual byte cases, geometric `Sizes`, and overflow-checked `LinearSizes`; comparator accepts explicit sizes and linear/geometric sweep flags. |
| `-n`, `-m`, `-w`, `-c` | Iterations, aggregation, warmup, checks implemented; printed controls matched in comparator. |
| `-N`, `-p` | Finite single-case cycles reuse communicators and allocations; host tests and 57 single-rank matched cycle cases pass. Shared size sweeps pass 114 matched single-rank samples. Context cancellation and continuous CLI SIGINT sweeps pass single-rank ordinary/graph tests; parallel initialization remains open (CYCLES.md). |
| `-o` | All six reductions implemented for twelve datatypes; complete reduction matrices cover ten; FP8 average and premultiplied sum fail in both installed implementations. Floating products are bounded; integer arithmetic wraps at element width. Odd-rank premultiplication physically unmeasured. |
| `-d` | Signed/unsigned 8-, 32-, and 64-bit integers, float32, and float64 implemented. Half, bfloat16, and FP8 runner integration is implemented and passes local checks; codecs pass CUDA conformance. Native collective and omission tests pass; half/BF16 upstream matrices pass; FP8 matrices remain pending (SMALLFLOAT.md). Physical receipt scope is tracked in DGX.md. |
| `-r` | Runner and comparator accept a root. Nonzero-root physical coverage absent. |
| In/out-of-place columns | Implemented for reference-checked host paths. Reference N/A columns are not counted as passes. |
| `-z`, `-y` | Dedicated or default CUDA stream implemented; default-stream single-rank tests pass. Synchronized timed iterations (-z 3) pass host tests and 19 single-rank matched cases with aggregation four. Modes 1/2 absent; see SYNCHRONIZATION.md. |
| `-T` | Comparator enforces a 45-second child-process timeout. Runner itself cannot interrupt a blocking native call. |
| `-G` | Capture/replay and matching comparator controls implemented locally. Host tests and one-rank float/sum replay/omission tests pass; 19 matched graph cases retained. Broader graph configurations unmeasured; see GRAPH.md. |
| `-C`, `-S`, `-J` | JSON receipts include campaign and child-process start/end timestamps. Reference CPU timing and per-iteration timestamps absent. |
| `-a` | Mean/rank-zero/min/max CUDA-event duration across local device ranks implemented. Upstream -a combines host-thread timings instead; non-mean comparison is restricted to one rank. No timing-equivalence claim. |
| `-R` | Local buffer registration implemented with deregistration before communicator destruction and memory release; single-rank positive and disjoint-buffer omission tests pass. Symmetric registration absent. |
| `-x`, `-D`, `-V` | CTA policy, device implementations, and device CTA counts absent. |
| `-H` | Host RMA implementations absent. |
| `-M`, `-U` | Memory and NCCL tuning reports absent. |
| `-u` | Element offset implemented in runner and comparator, with matching upstream printed controls and allocation overflow checks. Host tests and physical offset3 width/omission tests pass; 19 single-rank float/sum upstream cases pass. |
| `-I`, `-K` | Per-iteration event samples and leading-sample exclusion absent. |
| `-h` | Go flag help available; command syntax intentionally follows Go tooling. |
| All-to-all-v environment controls | Default spread-one qualified. Optional generated spread passes host routing fixtures and 18 single-rank matched samples; distributed routing remains unmeasured. Matrix files, scale, and summary absent; see TRAFFIC.md. |
| Other `NCCL_TESTS_*` overrides | Comparator rejects them before launching: rank split, device selection, and other reference-only changes cannot silently invalidate matching. |

Each datatype must include native element widths in layout, byte offsets,
bandwidth, transfer buffers, collective arguments, and premultiplication scalars.
The checker must preserve integer bits and account for each floating format's
rounding; float32 conversion alone is insufficient. Known-vector codec tests,
corruption controls, and matched physical runs are required before marking a
datatype covered.

The one-GPU DGX cannot establish peer routing, inter-GPU arithmetic, or transport
equivalence. Repeating a device ordinal does not resolve this. Broader hardware
evidence remains open while implementation work can continue on this host.
