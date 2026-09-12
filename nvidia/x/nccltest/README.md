# Go NCCL tests

This package implements Go-native correctness tests and CUDA-event benchmarks
for the host operations in NVIDIA/nccl-tests. It uses the existing pure-Go CUDA
and NCCL bindings; it does not invoke nccl-tests to perform a Go operation.
`ncclcompare` separately runs both executables to retain comparison receipts.

The current implementation supports signed and unsigned 8-, 32-, and 64-bit
integers plus float32 and float64 for all-reduce, all-gather, broadcast,
reduce-scatter, reduce, all-to-all, distance-weighted all-to-all-v (default spread),
scatter, gather, ring send/receive, and hypercube all-gather. Reduction operations
support sum, product, minimum, maximum, average, and pre-multiplied sum (`mulsum`). `Options.Aggregation` groups
multiple collectives per timed iteration; zero selects one. CUDA-event time is
normalized by iterations times aggregation, matching upstream `-m`. `Options.Devices` assigns
one distinct local GPU to each rank. Hypercube requires a power-of-two rank count.

`Case.Layout` follows upstream count and 16-byte per-rank alignment rules.
Requests that round down to zero work fail. In-place all-to-all, all-to-all-v,
and send/receive are rejected because upstream does not validate those modes.
The bandwidth formulas match each upstream operation, including zero bus
bandwidth for single-rank all-reduce and zero hypercube traffic at one rank.
These reporting formulas do not prove physical network activity.

Each correctness pass resets the buffers, varies input with rank and position,
executes the operation, synchronizes, copies the result to the host, and compares
it with an independent Go oracle. Floating receive storage starts poisoned with
NaNs; integer storage starts with the complement of each expected output.
`TestMissingCollective` exercises that path on a GPU with the collective omitted;
it must report every receive element as wrong. Local tests also inject corruption
and permute rank blocks. Successful NCCL status alone is never a correctness pass.

Run from the `nvidia` module:

```sh
GOWORK=off go test ./x/nccltest/... ./nccl
GOWORK=off go vet ./x/nccltest/... ./nccl
GOWORK=off go build -o nccltest ./x/nccltest/cmd/nccltest
GOWORK=off go build -o ncclcompare ./x/nccltest/cmd/ncclcompare
```

On a CUDA host, explicitly require physical execution:

```sh
NCCLTEST_CUDA=1 GOWORK=off go test ./x/nccltest -run 'TestCollectives|TestMissingCollective' -count=1 -v
NCCLTEST_CUDA=1 GOWORK=off go test ./x/nccltest -run '^$' -bench BenchmarkCollectives -benchtime=20x -count=1
./nccltest -op all_reduce -bytes 65536 -devices 0 -warmup 5 -iterations 20 -checks 2
```

The explicit GPU tests fail on missing CUDA/NCCL rather than silently skipping.
Without `NCCLTEST_CUDA=1`, only the physical tests skip. Benchmarks report
`gpu-ns/op`, `alg-GB/s`, and `bus-GB/s`; they suppress host `ns/op`, which would
include communicator initialization and correctness checks.

For matched comparisons, build pinned nccl-tests revision
`b4d5beebca8a76cf01335f724d154b9b9d394d96`, then use a new receipt directory:

```sh
./ncclcompare -reference /path/to/nccl-tests/build -go-test "$PWD/nccltest" -output "$HOME/tmp/nccl-comparison"
./ncclcompare -reference /path/to/nccl-tests/build -go-test "$PWD/nccltest" -aggregation 4 -sizes 65536 -reductions sum -output "$HOME/tmp/nccl-aggregation"
./ncclcompare -reference /path/to/nccl-tests/build -go-test "$PWD/nccltest" -datatype double -output "$HOME/tmp/nccl-double"
```

The default matrix includes all eleven operations, six reductions where
applicable, 16/1024/65536-byte sizes, and every mode upstream checks. It requires
matching effective byte counts, element counts, root, reduction, NCCL runtime version, and printed thread/GPU/warmup/iteration/aggregation/check
controls. Every child has a 45-second timeout. Raw output, command lines, executable
SHA-256 hashes, wrong counts, and timing ratios are retained. Upstream `N/A`
columns are excluded from checked coverage. Truncated output, failed process
status, zero comparisons, or missing completion summaries cannot pass.
The comparator also rejects inherited `NCCL_TESTS_*` overrides, which can change
only the reference's devices, rank groups, or traffic. Unset them before running
a matched comparison. The Go result must report the exact requested case.

## Scope still to close

See [COVERAGE.md](COVERAGE.md) for the source-checked option inventory and
remaining implementation and physical-validation requirements.
The remaining floating-format codecs have independent CUDA conformance checks;
[SMALLFLOAT.md](SMALLFLOAT.md) describes their implemented runner integration and pending GPU validation.

This is an active implementation, not a claim of full nccl-tests equivalence.
GPU validation of half, bfloat16, and FP8, traffic matrices/spread selection, and coverage of advanced graph, registration,
device-kernel, RMA, MPI, and multi-thread modes remain to be assessed or implemented.

The qualified DGX has one GB10 GPU. Its single-rank runs can check ABI calls,
local buffer semantics, arithmetic identity cases, and reporting conventions.
They cannot prove inter-GPU reduction, peer routing, or network transport. CPU
multi-rank oracle tests are separate evidence and do not fill that hardware gap.
Timing ratios are observations; they are not claims of equal performance.


For `mulsum`, the scalar matches upstream: 1 on even communicator ranks and 2
on odd ranks. The host scalar is pinned across its immediate-copy FFI call.
The custom operator is created per collective and destroyed when that
collective API returns, including within an outer aggregation group.

`Case.DataType` defaults to `Float32`; select `Float64` or pass `-datatype double`
to either command for doubles. Layout counts are elements of the selected type.
`Check` accepts float32 output, and `Check64` accepts float64 output. Double
payloads include fractions below float32 precision to detect accidental
narrowing. The floating checkers use relative tolerances of 1e-5 and 1e-12,
respectively, with an absolute floor of the same value. A nonfinite expected
value is an error. Floating products use four magnitude-bearing ranks and
rank- and position-dependent signs on later ranks, keeping products below
14^4 for all supported communicator sizes.

Integer datatypes are `int8`, `uint8`, `int32`, `uint32`, `int64`, and `uint64`.
`CheckBytes` accepts native-endian buffers for any supported type and checks
integer values bit for bit. Integer payloads exercise all bits, including low
bits of 64-bit values above float64's exact range. Sum and product wrap at the
element width; integer average divides the wrapped sum, truncating signed
results toward zero, as in the pinned reference.

The comparator accepts `-root` for rooted collectives and rejects roots outside
the communicator before launching either child. Physical nonzero-root coverage
requires a host with more than one GPU.

### Message-size sweeps

The comparator accepts explicit `-sizes 16,1024,65536`, or a sweep:
`-first 16 -last 65536 -step 16` for linear sizes, or
`-first 16 -last 65536 -factor 2` for geometric sizes. Each generated size
runs both executables with the same byte request. Mixing explicit sizes with
sweep flags, duplicate sizes, and invalid increments is rejected before any
child process starts. Library callers can use `LinearSizes` or `Sizes`.

Both commands accept `-timing mean|rank0|min|max`. Go selects across local
device CUDA-event durations. Upstream `-a` selects across host-thread wall
timings, so these are different measurement populations. The comparator limits
non-mean selection to one rank and passes the corresponding upstream flag.
Correctness comparisons do not establish equal timing or performance.

`-default-stream` uses CUDA stream zero in both executables (reference `-y 1`).
The runner synchronizes this stream during cleanup but never destroys it.
Physical validation of this option remains pending.

`-register` enables NCCL local registration for both allocated buffers (reference
`-R 1`). Registrations are released before communicators and device memory,
including when later setup fails. This option has not yet been exercised on
the DGX; it does not claim symmetric registration or a performance benefit.

The comparator atomically checkpoints `comparison.json` after each checked case.
`RequestedWorkloads` and `FinishedWorkloads` count operation/reduction/size
combinations before in-place expansion. `Complete` stays false until all
requested workloads pass. Interrupted receipts
can retain useful completed cases but must not be treated as complete matrices.
Candidate timing must be positive and bandwidth finite and nonnegative; zero
bandwidth is valid for single-rank operations with no transferred traffic.
