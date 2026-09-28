# Device debugger qualification

This probe compiles a materialized CUDA source with `WithName` and
`WithDeviceDebug`, loads it through the managed module API, and launches it from
Go. `TestDeviceDebugger` runs the probe under the actual `cuda-gdb`, stops at a
source breakpoint, inspects CUDA threads and two variables, then checks the
copied-back output and normal exit.

From the nvidia module, inside an exclusive GPU reservation:

```sh
GOWORK=off NVRTCX_DEBUGGER_TEST=1 go test -count=1 -v \
  -run '^TestDeviceDebugger$' ./x/nvrtcx
```

With the flag enabled, missing tooling or a failed breakpoint fails the test.
Without it, the test explicitly skips. The outer GPU reservation must cover the
entire test including the debugger subprocess. The test has a 90-second deadline.

This qualifies device debugging of a Go-launched NVRTC kernel. It does not qualify
Go goroutine inspection, optimized variable availability, cudabind-generated
source materialization, or multi-stream/graph debugging. Device debug disables
optimization; use default line information for profiling production kernels.
The test does not change debugger auto-load security settings or system config.
