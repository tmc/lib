# GoMLX through TMC PJRT

This optional module implements `github.com/gomlx/compute` using the TMC
StableHLO builder and PJRT runtime. Native execution does not require cgo.

```go
backend, err := gomlx.New(pluginPath)
if err != nil {
    return err
}
defer backend.Close()
```

Pass the backend to GoMLX `graph.NewExec` or `model.NewExec`. Call `Close`
explicitly when cleanup errors matter. Runtime calls are safe for concurrent
use and serialize native access. Each graph builder has one owner.

Current support is static Bool, F32, Int32, Int64, Uint32, and Uint64 on the first exposed device: constants,
parameters, Add, Sub, Mul, Div, ConvertDType, Identity, F32 Sqrt, Pow, Exp, Log, Neg, Abs, and Cos, Min and Max, BroadcastInDim,
Reshape, Transpose, numeric Iota, ReduceSum, ReduceMax,
comparisons (Equal, LessThan, and LessOrEqual), LogicalAnd,
Where, Philox RNGBitGenerator with a [3]uint64 state, and DotGeneral with
default configuration. Named axes, other dtypes, non-default dot configuration,
subfunctions, sharding, and remaining operations return unsupported errors.
Donation invalidates inputs without reusing native storage.
Bool is qualified for constants, transfers, shape operations, comparisons,
LogicalAnd, and Where.
Where broadcasts scalar operands; other operand shapes must match.

Native tests cover Philox known answers and state progression, GoMLX uniform
and normal random graphs, arithmetic, broadcasting, contractions, GoMLX graphs,
boolean transfers and conditional selection, masked softmax,
GoMLX attention.Core in BHSD/BSHD layouts with grouped-query and causal masking,
attention gradients checked against independent host finite differences,
softmax and log-softmax with cross-entropy gradients, autodiff, stateful gradient-descent updates, built-in SGD with decay, Adam with moment-state checkpoint
recovery, end-to-end Dense/RMSNorm/causal-attention training with full Adam
checkpoint recovery, ownership,
and concurrent runtime access. This is not full GoMLX backend conformance.

Run tests with `TMC_COMPUTE_PLUGIN` set to a compatible PJRT CPU plugin:

```
GOWORK=off CGO_ENABLED=0 go test -count=1 ./...
GOWORK=off CGO_ENABLED=1 go test -race -run 'TestNativeConcurrentRuntime|TestNativeBackendFinalize' ./...
```

The second command requires cgo for Go's race detector; it does not instrument
the native plugin. Without the environment variable, native tests skip.
Sibling TMC modules use relative replacements, following this repository's
module layout. Upstream GoMLX dependencies use pinned downloadable versions.

## Transfer allocation profiles

`BenchmarkTransfer` measures scalar and vocabulary-sized F32 upload/download.
Upload includes buffer finalization. Download uses a preallocated caller
slice; the backend's temporary readback storage is included in the result.
Setup checks a nonzero input pattern and rejects a wrong-result control.

```
CGO_ENABLED=0 go test -run '^$' -bench BenchmarkTransfer -benchmem -count=5
CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkTransfer/logits/download$' -benchtime=100x -memprofilerate=1 -memprofile="$HOME/tmp/gomlx.pprof" -o "$HOME/tmp/gomlx.test"
go tool pprof -sample_index=alloc_space "$HOME/tmp/gomlx.test" "$HOME/tmp/gomlx.pprof"
```

Heap profiles include setup. Filter measured benchmark stacks before assigning
costs to the timed operation. Go heap totals exclude native/device allocations;
profiled timings are not latency comparisons.
