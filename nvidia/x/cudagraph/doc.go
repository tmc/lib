// Package cudagraph provides an idiomatic Go interface for building, capturing,
// instantiating, and launching CUDA Graphs.
//
// CUDA Graphs eliminate CPU launch overhead by defining dependency DAGs of kernel launches
// and memory copies once, validating execution topologies upfront, and replaying them
// with sub-microsecond latency.
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
