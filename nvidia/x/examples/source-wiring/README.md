# source-wiring

Follows kernels written in Go down to the instructions the GPU executes.

Four kernels are built with `cudatensor`, lowered to CUDA C++ by `cudafuse`,
and attributed by `cudalineinfo`, which compiles them with NVRTC and
disassembles them with `nvdisasm`. The program reports, per Go statement, how
many generated CUDA lines, PTX locations, and SASS instructions name it, then
lists the first kernel instruction by instruction. From the `nvidia` module
directory:

    go run ./x/examples/source-wiring -arch sm_121

`nvdisasm` must understand the target architecture. A CUDA 12.x `nvdisasm` fails
on `sm_121` with "does not contain Cuda architecture information"; pass a newer
one with `-nvdisasm /usr/local/cuda-13.0/bin/nvdisasm`.

## What it shows

The link is carried by `#line` directives that `cudafuse.Fuse` emits from the
source map it already builds. NVRTC records them as ordinary line information,
so every tier below Go inherits Go's coordinates. `cudalineinfo.Attribute`
joins the three tiers on the Go file and line.

    fused_gate: 162 of 464 SASS instructions name a program line
      SITE           SOURCE                                                   CUDA  PTX  SASS
      kernels.go:12  x := cudatensor.NewInput(g, "x", shape, cudair.Float32)  2     5    14
      kernels.go:13  w := cudatensor.NewInput(g, "w", shape, cudair.Float32)  2     5    12
      kernels.go:14  act := x.Silu()                                          1     5    117
      kernels.go:15  return act.Mul(w)                                        3     5    19
      remainder: 125 generated, 0 unlocated, 62 in $__internal_0_$__cuda_sm20_div_u64, 115 in $__internal_1_$__cuda_sm3x_div_rn_noftz_f32_slowpath

The kernel's grid-stride loop is unrolled five times, so each Go line appears
in five PTX locations and its instructions come in five copies. The
instruction listing is the part worth reading, because it can be checked
against the Go line by eye (one copy of each line, abbreviated):

      kernels.go:14  act := x.Silu()
          FFMA.SAT                 R18, R0, -R19, 0.5
          MUFU.EX2                 R19, R19
          FFMA                     R21, R18, R19, 1
          MUFU.RCP                 R3, R21
          FCHK                     P0, R0, R21
          @!P0 BRA                 `(.L_x_6)
          CALL.REL.NOINC           `($__internal_1_$__cuda_sm3x_div_rn_noftz_f32_slowpath)
      kernels.go:15  return act.Mul(w)
          IADD.64                  R16, R14, UR6
          FMUL                     R3, R2, R3
          STG.E                    desc[UR28][R16.64], R3

`Silu` is `x / (1 + expf(-x))`, and its instructions are an exponential, a
reciprocal refined by Newton steps, a range check, and a call to the division
slowpath for inputs the check rejects. `Mul` is one `FMUL`, the address, and
the store.

## Through helpers

The last kernel builds its graph through two helper functions, so the lines
that own the operations are inside `activate` and `normalize`, not in the
kernel body a reader looks at. Every captured location also records the Go
frames above it, and the roll-up charges instructions to those frames the way
a profiler's cumulative column does:

    fused_layered: SASS instructions by frame, cumulative
      SITE           FUNCTION        SOURCE                                                      FLAT  CUM
      kernels.go:38  main.layered    x := cudatensor.NewInput(g, "x", shape, cudair.Float32)     14    14
      kernels.go:39  main.layered    w := cudatensor.NewInput(g, "w", shape, cudair.Float32)     12    12
      kernels.go:40  main.layered    act := activate(x)                                          0     115
      kernels.go:41  main.layered    return normalize(act, w)                                    0     159
      kernels.go:45  main.activate   return t.Silu()                                             115   115
      kernels.go:49  main.normalize  scale := w.Sqrt()                                           70    70
      kernels.go:50  main.normalize  return t.Div(scale)                                         89    89
      main.go:72     main.main       if err := run(*arch, *nvdisasm, *units, cfg); err != nil {  0     300
      main.go:81     main.run        a, fk, err := attribute(ctx, k, opts)                       0     300
      main.go:121    main.attribute  k.build(g)                                                  0     300

The two call lines own no instructions of their own and account for all of
them. The frames continue up to the driver, where the line that built the
graph carries the whole kernel.

## At run time

The tables above are static. With `-pcsampling` the first kernel also runs
under CUPTI's PC sampler, and with `-sassmetrics` under its instruction
patcher, and each measure is joined to the same Go lines:

    go run ./x/examples/source-wiring -pcsampling -sassmetrics -pprof gate.pb.gz

    fused_gate: 31498 samples by program line
      SITE           SOURCE                                                   SASS  SAMPLES  SHARE
      kernels.go:12  x := cudatensor.NewInput(g, "x", shape, cudair.Float32)  14    469      1.5%
      kernels.go:13  w := cudatensor.NewInput(g, "w", shape, cudair.Float32)  12    202      0.6%
      kernels.go:14  act := x.Silu()                                          117   13593    43.2%
      kernels.go:15  return act.Mul(w)                                        19    1639     5.2%
      elsewhere: 15595 at instructions naming no program line, 0 at unresolved PCs

    fused_gate: 78643200 executed by program line
      SITE           SOURCE                                                   SASS  EXECUTED  SHARE
      kernels.go:12  x := cudatensor.NewInput(g, "x", shape, cudair.Float32)  14    1966080   2.5%
      kernels.go:13  w := cudatensor.NewInput(g, "w", shape, cudair.Float32)  12    1310720   1.7%
      kernels.go:14  act := x.Silu()                                          117   13107200  16.7%
      kernels.go:15  return act.Mul(w)                                        19    2621440   3.3%
      elsewhere: 59637760 at instructions naming no program line, 0 at unresolved PCs

Samples are where the warps were, so they weight `Silu` by its latency: 43%
of the time on a quarter of the static instructions. Executed counts are
exact. Twenty launches of 2^20 elements are 655360 warp launches, one element
per thread, so only the first of the five unrolled copies runs: the `x` input
line ran 3 × 655360 instructions, `Mul` 4 × 655360, and `Silu` 20 × 655360,
its copy of 23 less the `MOV`, `CALL` and `MOV` around the slowpath call,
which never executed because no input took the slow division path. The static
count says what the line became; the executed count says which of it ran.

Most of what ran names no Go line: 91 instructions per warp of index
arithmetic, the launch-shape guard, and the 64-bit division that computes
the unrolled loop's trip count, a call into `$__cuda_sm20_div_u64`. That is the cost of the
kernel's generated scaffolding, and for a kernel this small it outweighs the
Go statements three to one.

`-pprof` writes the samples as a pprof profile. Each sample's stack runs from
the SASS instruction through the PTX and generated CUDA lines to the Go
statement and the frames that called it, so `go tool pprof -lines` reads the
kernel by Go line and `-focus=main.gate` filters to one kernel's builder.

## What the collectors cost

Both collectors time their own launches, so the run ends with the wall time
of the same work under each. On a GB10 (`sm_121`, CUDA 13.4, driver 580.173),
median per-launch time of five runs:

    go run ./x/examples/source-wiring -pcsampling -sassmetrics
    go run ./x/examples/source-wiring -pcsampling -sassmetrics -n 16777216 -launches 100 -period 16

| workload | none | -pcsampling | -sassmetrics |
|---|---|---|---|
| 2^20 elements, 20 launches, period 10 | 0.018 ms | 0.041 ms (+128%) | 8.98 ms (×499) |
| 2^24 elements, 100 launches, period 16 | 0.895 ms | 0.932 ms (+4.1%) | 144.6 ms (×162) |

PC sampling costs 23 to 37 microseconds per launch, which is the host
draining the hardware buffer between launches. That is 4% of a kernel that
runs for 0.9 ms and more than double the cost of one that runs for 18
microseconds, so the overhead is a property of the launch, not of the kernel.
The instruction patcher rewrites the kernel and is two orders of magnitude
more expensive; it buys exact counts rather than a sample.

The sampler's hardware buffer is fixed, so a short period over a long
workload overflows it, which CUPTI reports as running out of memory. The
run then retries with the period lengthened by two, each retry sampling a
quarter as often, and says which period the numbers came from:

    $ go run ./x/examples/source-wiring -pcsampling -n 16777216 -launches 20 -period 10
      sampling period raised from 2^10 to 2^14 cycles: the hardware buffer overflowed
    fused_gate: 136808 samples by program line

At 2^24 elements the default period of 10 does not survive 20 launches and
14 is the shortest that does.

Both collectors need CUPTI 13 and exclusive use of the device's profiling
hardware, so neither is on by default.

## Fast math

The first kernel is recompiled with `-use_fast_math` and the two builds are
diffed per Go line, as multisets of opcodes:

    fused_gate: default (162 of 464) vs -use_fast_math (69 of 264)
      SITE           DEFAULT  -USE_FAST_MATH  REMOVED                                                           ADDED
      kernels.go:12  14       14              LDCU                                                              BSSY
      kernels.go:13  12       12              -                                                                 -
      kernels.go:14  117      25              BRA×5 BSSY×5 BSYNC×5 CALL×5 FCHK×5 FFMA×50 HFMA2×5 IMAD×5 MOV×17  FMUL×10
      kernels.go:15  19       18              LDCU                                                              -
      remainder: default: 125 generated, 0 unlocated, 62 in $__internal_0_$__cuda_sm20_div_u64, 115 in $__internal_1_$__cuda_sm3x_div_rn_noftz_f32_slowpath
      remainder: -use_fast_math: 124 generated, 0 unlocated, 71 in $__internal_0_$__cuda_sm20_div_u64
      max |fast - default| over 4096 elements: 7.63e-06 (max |default| 110)

The flag's work is on one Go line. In each unrolled copy `Silu` lost its IEEE
division (the `FCHK` range check, the `CALL` into the slowpath, and the Newton
iterations) for two `FMUL`s around the reciprocal, and the float slowpath
subroutine left the binary with it. The input and output lines changed only
in where the scheduler put a constant load. The integer division that sizes
the loop is not floating point, so it stays. Both builds then run on the same inputs, so the instruction diff and
its cost in accuracy are read together.

## Afterwards

`-units dir` writes each kernel's unit, annotated source, PTX, CUBIN, and
disassembly to a directory, and the `cudalineinfo` command asks the same
questions of the artifacts without rebuilding them:

    go run ./x/examples/source-wiring -units /tmp/units
    go run ./x/cmd/cudalineinfo -sass /tmp/units/fused_layered.sass -format rollup /tmp/units/fused_layered.json

`-format digraph` prints the unit as a call graph in the form
`golang.org/x/tools/cmd/digraph` reads: program functions down to the line
that built each operation, the line to the kernel, and lines to the SASS
subroutines their instructions call. Asking which Go path pulled the
division slowpath into `fused_layered`:

    $ go run ./x/cmd/cudalineinfo -sass /tmp/units/fused_layered.sass -format digraph /tmp/units/fused_layered.json |
        digraph somepath main.main '$__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath'
    main.main main.run
    main.run main.attribute
    main.attribute main.layered
    main.layered main.activate
    main.activate kernels.go:45
    kernels.go:45 $__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath

The path ends at `return t.Silu()`, whose sigmoid divides. The JSON unit is
what carries the functions and callers; the annotated `.cu` alone gives the
lines and the kernel.

The functions are named the way the runtime names them, so the graph
composes with the program's own call graph. `callgraph` names a function by
its package path, which for a main package is the import path rather than
`main`, so one rename joins the two and the composed graph answers for
paths no run recorded:

    $ { callgraph -algo=static -format=digraph ./x/examples/source-wiring |
          sed 's|github.com/tmc/lib/nvidia/x/examples/source-wiring\.|main.|g'
        go run ./x/cmd/cudalineinfo -sass /tmp/units/fused_layered.sass \
          -format digraph /tmp/units/fused_layered.json; } |
        digraph somepath main.fastMath '$__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath'
    main.fastMath main.attribute
    main.attribute main.layered
    main.layered main.activate
    main.activate kernels.go:45
    kernels.go:45 $__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath

The static half supplies calls the unit never saw; the compiled half
supplies what those calls became.

`cudaprof lines` asks the same question of a capture. A capture made with
`cudaprof capture --inject cupti_agent.so -- source-wiring` stores each
module's CUBIN, and the view disassembles them and prints one table per
module, so the default and fast-math builds of `fused_gate` come out
separately. A capture that records PC samples with source frames, as the
`-pprof` profile does, fills the sample column per line.

The program also publishes into the capture what only it has: each kernel's
unit, which is what lets the tables name Go lines and roll them up to their
callers, and, with `-sassmetrics`, the exact executed counts, which become the
capture's `executed` sample type next to the sampled one.

    cudaprof lines -sample executed capture.pprof
    cudaprof lines -format rollup capture.pprof
    cudaprof lines -sample pc_samples -diff other.pprof fused_gate capture.pprof

The last compares two captures line by line: the instructions each build gave
a line, and what each run measured there.

On a GB10 the executed counts do not reach a capture. While the injected
agent holds activity tracing, CUPTI returns zero for every patched counter,
so `-sassmetrics` under `cudaprof capture` prints an all-zero table and the
capture has no `executed` sample type. The in-process PC sampler does run
alongside the agent (32976 samples, half of them on `kernels.go:14`), but its
samples go to the table and `-pprof`, not into the capture.

## The control

Withholding the `#line` directives and recompiling the same graph must remove
the attribution entirely:

    control (fused_gate): SASS instructions naming a Go file
      with #line directives:    162 of 464
      without #line directives: 0 of 464

The program exits non-zero if the control arm attributes anything, or if the
directive arm attributes nothing.

## What the numbers do not say

The counts are honest about two things that are easy to misread.

`162 of 464` is not 35% coverage of the program's work. The remainder line
accounts for the rest: the loop's index arithmetic and bounds checks on
generated lines, and the compiler's own subroutines for the 64-bit trip-count
division and the float division slowpath, which belong to no line of user Go.

A statement can reach zero. In `fused_rescale`, `scaled := v.Div(inv)` gets
`PTX 0, SASS 0` while the following `Neg()` gets 84 instructions: the compiler
fused the divide into its consumer and attributed the result to one line. That
is the optimizer being visible through the mapping, not the mapping failing.
A Go statement is not guaranteed to survive as instructions of its own.
