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
// [Capture] records operations enqueued by its callback on the supplied stream.
// The caller must handle operation errors inside that callback; Capture accepts
// func(), so it cannot return those errors on the caller's behalf.
//
// After a successful capture, release the graph with [Graph.Destroy]. Create
// an executable with [Graph.Instantiate], check its error, and release that
// executable with [Exec.Destroy]. Check [Exec.Launch] and synchronize the stream
// before releasing resources still used by queued work.
package cudagraph
