// Package cudagraph provides an experimental Go interface for building,
// capturing, instantiating, and launching CUDA Graphs.
//
// CUDA Graphs define dependency DAGs of kernel launches and memory copies once
// and replay the instantiated graph. Whether replay reduces launch overhead is
// workload- and device-dependent; this package does not promise a latency bound.
// The repository does not currently use this package in a production path.
//
// # Stream Capture
//
// The simplest way to create a CUDA graph is by recording existing stream operations:
//
//	g, err := cudagraph.Capture(stream, func() {
//		launcher.On(stream).Launch(...)
//		sliceB.CopyDeviceToDeviceAsync(sliceA, stream)
//	})
//	exec, err := g.Instantiate()
//	defer exec.Destroy()
//
//	err = exec.Launch(stream)
package cudagraph
