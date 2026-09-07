# XLA

This module provides Go ownership wrappers for PJRT plugins, compiled programs
and device buffers. It uses `openxla` bindings and builds without cgo.

Close buffers and executables before their engine. Execution waits for native
completion, including when its context is cancelled. Closing a resource must
be synchronized with calls that use it. A destruction error invalidates the Go
handle; it does not promise that the native runtime released the resource.

Uploads validate the byte length against the element type and dimensions.
Downloads validate destination size. Zero compile-option counts mean one;
negative counts are rejected. The compile-options encoder has no protobuf
library dependency.

Run `go test ./...` for host tests. To test the complete Go-to-PJRT path with a
CPU plugin installed separately:

```
XLA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so go test -run TestNativePJRT -v
CGO_ENABLED=0 XLA_PJRT_CPU_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so go test -run TestNativePJRT -v
```

The test checks compilation, host uploads, execution, output metadata,
downloads and resource cleanup using an asymmetric matrix multiplication. It
is skipped when the environment variable is unset. It does not install a
plugin or validate TPU/GPU execution.
