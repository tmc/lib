# source-wiring

Follows kernels written in Go down to the instructions the GPU executes.

Four kernels are built with `cudatensor`, lowered to CUDA C++ by `cudafuse`,
and attributed by `cudalineinfo`, which compiles them with NVRTC and
disassembles them with `nvdisasm`. The program reports, per Go statement, how
many generated CUDA lines, PTX locations, and SASS instructions name it, then
lists the first kernel instruction by instruction. From the `nvidia` module
directory:

    go run ./x/examples/source-wiring -arch sm_121

`nvdisasm` must understand the target architecture; an older one fails with
"does not contain Cuda architecture information". Pass a newer one with
`-nvdisasm path`.

## What it shows

The link is carried by `#line` directives that `cudafuse.Fuse` emits from the
source map it already builds. NVRTC records them as ordinary line information,
so every tier below Go inherits Go's coordinates. `cudalineinfo.Attribute`
joins the three tiers on the Go file and line.

Each kernel gets a table. The output looks like this; the counts vary with
the toolchain and GPU:

    fused_gate: 32 of 184 SASS instructions name a program line
      SITE           SOURCE                                                   CUDA  PTX  SASS
      kernels.go:12  x := cudatensor.NewInput(g, "x", shape, cudair.Float32)  2     1    4
      kernels.go:13  w := cudatensor.NewInput(g, "w", shape, cudair.Float32)  2     1    2
      kernels.go:14  act := x.Silu()                                          1     1    22
      kernels.go:15  return act.Mul(w)                                        3     1    4
      remainder: 31 generated, 0 unlocated, 121 in $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath

CUDA, PTX, and SASS count the generated CUDA lines, PTX locations, and SASS
instructions that name the Go line. The remainder line accounts for the
instructions that name no Go line: those on generated lines (index
arithmetic, bounds checks), those with no line at all, and those inside
compiler subroutines such as the float division slowpath.

The listing of the first kernel is the part worth reading, because it can be
checked against the Go line by eye. `Silu` is `x / (1 + expf(-x))`, and its
instructions are an exponential, a reciprocal refined by Newton steps, a
range check, and a call to the division slowpath for inputs the check
rejects. `Mul` is one `FMUL`, the address, and the store.

## Through helpers

The last kernel builds its graph through two helper functions, so the lines
that own the operations are inside `activate` and `normalize`, not in the
kernel body a reader looks at. Every captured location also records the Go
frames above it, and the roll-up charges instructions to those frames the way
a profiler's cumulative column does. FLAT is the instructions a line owns;
CUM adds those of every line it called.

The two call lines in `layered` own no instructions of their own and account
for all of the helpers' instructions. The frames continue up through the
driver in `main.go`, where the line that built the graph carries the whole
kernel.

## At run time

The tables above are static. With `-pcsampling` the first kernel also runs
under CUPTI's PC sampler, and with `-sassmetrics` under its instruction
patcher, and each measure is joined to the same Go lines:

    go run ./x/examples/source-wiring -pcsampling -sassmetrics -pprof gate.pb.gz

`-n` sets the elements per launch and `-launches` the number of launches.
Each table has the same shape: SASS is the line's static instruction count,
SAMPLES or EXECUTED the measure, SHARE its fraction of the total, and an
`elsewhere` line counts what landed on instructions naming no Go line.

Samples and executed counts answer different questions. Samples are where
the warps were, so they weight a line by its latency: a line with slow
instructions takes more samples than its instruction count suggests.
Executed counts are exact, counted per warp: a line's static count says what
the line became, its executed count says which of it ran. Instructions that
never run, such as the division slowpath call when no input takes the slow
path, have a static count and no executed count.

What executes outside any Go line is the kernel's generated scaffolding:
index arithmetic and the launch-shape guard. For a kernel this small it is a
large part of the work.

`-pprof` writes the samples as a pprof profile. Each sample's stack runs from
the SASS instruction through the PTX and generated CUDA lines to the Go
statement and the frames that called it, so `go tool pprof -lines` reads the
kernel by Go line and `-focus=main.gate` filters to one kernel's builder.

## What the collectors cost

Both collectors time their own launches, so the run ends with the wall time
of the same work under each, per launch and as overhead against the
uninstrumented run.

PC sampling costs a roughly fixed time per launch, which is the host draining
the hardware buffer between launches. Its relative overhead is therefore a
property of the launch, not of the kernel: small for a long kernel, large for
a short one. The instruction patcher rewrites the kernel and is far more
expensive; it buys exact counts rather than a sample.

The sampler's hardware buffer is fixed, so a short period over a long
workload overflows it, which CUPTI reports as running out of memory. The
run then retries with the period lengthened by two, each retry sampling a
quarter as often, and says which period the numbers came from:

    $ go run ./x/examples/source-wiring -pcsampling -n 16777216 -launches 20 -period 10
      sampling period raised from 2^10 to 2^14 cycles: the hardware buffer overflowed

`-period` sets the exponent to start at, one sample per 2^n cycles.

Both collectors need CUPTI 13 or later and exclusive use of the device's
profiling hardware, so neither is on by default.

## Fast math

The first kernel is recompiled with `-use_fast_math` and the two builds are
diffed per Go line, as multisets of opcodes: the instruction count of each
build, and the opcodes the flag removed and added. Both builds then run on
the same inputs, and the run reports the largest difference between their
results, so the instruction diff and its cost in accuracy are read together.

The flag's work lands on one Go line. `Silu` loses its IEEE division (the
`FCHK` range check, the `CALL` into the slowpath, and the Newton iterations)
for multiplications around the reciprocal, and the slowpath subroutine
leaves the binary with it. The input and output lines change little or not
at all.

## Afterwards

`-units dir` writes each kernel's unit, annotated source, PTX, CUBIN, and
disassembly to a directory, and the `cudalineinfo` command asks the same
questions of the artifacts without rebuilding them:

    go run ./x/examples/source-wiring -units /tmp/units
    go run ./x/cmd/cudalineinfo -sass /tmp/units/fused_layered.sass -format rollup /tmp/units/fused_layered.json

`-format digraph` prints the unit as a call graph in the form
`golang.org/x/tools/cmd/digraph` reads: program functions down to the line
that built each operation, the line to the kernel, and lines to the SASS
subroutines their instructions call. The subroutines are named as the
disassembly names them, with a numbered prefix. Asking which Go path pulled
the division slowpath into `fused_layered`:

    $ go run ./x/cmd/cudalineinfo -sass /tmp/units/fused_layered.sass -format digraph /tmp/units/fused_layered.json >layered.digraph
    $ digraph somepath main.main "$(grep -o '[^ ]*div_rn_noftz_f32_slowpath' layered.digraph | head -1)" <layered.digraph
    main.main main.run
    main.run main.attribute
    main.attribute main.layered
    main.layered main.activate
    main.activate kernels.go:45
    kernels.go:45 $__internal_1_$__cuda_sm3x_div_rn_noftz_f32_slowpath

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
        cat layered.digraph; } |
        digraph somepath main.fastMath "$(grep -o '[^ ]*div_rn_noftz_f32_slowpath' layered.digraph | head -1)"
    main.fastMath main.attribute
    main.attribute main.layered
    main.layered main.activate
    main.activate kernels.go:45
    kernels.go:45 $__internal_1_$__cuda_sm3x_div_rn_noftz_f32_slowpath

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

The executed counts may not reach a capture. If CUPTI returns zero for every
patched counter while the injected agent holds activity tracing,
`-sassmetrics` under `cudaprof capture` prints an all-zero table and the
capture has no `executed` sample type. The in-process PC sampler does run
alongside the agent, but its samples go to the table and `-pprof`, not into
the capture.

## The control

Withholding the `#line` directives and recompiling the same graph must remove
the attribution entirely. The run prints the number of SASS instructions
naming a Go file with and without the directives, and exits non-zero if the
control arm attributes anything, or if the directive arm attributes nothing.

## What the numbers do not say

The counts are easy to misread in two ways.

The fraction of instructions that name a program line is not coverage of the
program's work. The remainder line accounts for the rest: the loop's index
arithmetic and bounds checks on generated lines, and compiler subroutines
such as the float division slowpath, which belong to no line of user Go.

A statement can reach zero. When the compiler fuses an operation into its
consumer, it attributes the result to one line, and the other line gets
`PTX 0, SASS 0`. That is the optimizer being visible through the mapping, not
the mapping failing. A Go statement is not guaranteed to survive as
instructions of its own.
