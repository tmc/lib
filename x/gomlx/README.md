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

Current support is static F32 on the first exposed device: constants,
parameters, Add, Mul, BroadcastInDim, Reshape, ReduceSum, and DotGeneral with
default configuration. Named axes, other dtypes, non-default dot configuration,
subfunctions, sharding, and remaining operations return unsupported errors.
Donation invalidates inputs without reusing native storage.

Native tests cover arithmetic, broadcasting, contractions, GoMLX graphs,
autodiff, stateful gradient-descent updates, checkpoint recovery, ownership,
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
