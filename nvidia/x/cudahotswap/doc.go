// Package cudahotswap checks and replaces CUDA kernels in a retained service.
//
// Register retains an original kernel. Push compiles and checks the actual PTX
// ABI, runs configured replay and application checks, and publishes a new
// generation with rollback on ordinary application failure. Select and Revert
// switch retained code. SetWatchPaths and Watch push source and tracked header
// changes through the same checks. Subscribe receives bounded report events.
//
// Use Launch and Wait, or Pin and Release, to keep module ownership through
// actual completion. RegisterGraph and LaunchGraph retain and patch graph
// generations. Raw functions used outside these paths bypass lifetime guards.
// Close must succeed before the caller closes its buffers, graphs and device.
// Cancellation alone never establishes completion or permits native teardown.
//
// Device faults and timeouts poison the service; the owning process decides
// whether to restart. Host controls and compilation do not establish native
// execution, and request wall latency is separate from native CUDA event time.
package cudahotswap
