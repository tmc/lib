// Package cuptibenchcapture packages a cuptibench counter profile together with
// the exact code that produced it, as a portable capture cmd/cudaprof can open.
//
// A cuptibench run exports a pprof profile and a JSON file of results beside
// it. Neither carries the compiled code, so a reader cannot get from a counter
// back to a source line. Compose attaches the source, PTX, CUBIN and SASS of
// the module the benchmark loaded, records the module generation by digest, and
// keeps the counter capture and the separately timed execution distinct.
//
// Compilation, disassembly and viewing stay here, in the consumer. Nothing in
// this package runs on the GPU: it takes bytes a benchmark already produced.
package cuptibenchcapture
