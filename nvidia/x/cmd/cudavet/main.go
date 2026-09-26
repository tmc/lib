// Command cudavet reports CUDA hazards that a Go compiler cannot see.
//
// Checks:
//
//	cargalign          kernel parameter layouts that disagree with cudakernel.Schema
//	unpinned           Go memory handed to asynchronous DMA without a runtime.Pinner
//	leak               handles created and never released
//
//	contextthread      a context made current without runtime.LockOSThread
//	unlockdestroy      a context released on a different thread from the one it is current on
//	graphshare         a graph used concurrently from two goroutines
//
// Usage:
//
//	cudavet [flags] package...
//
// The checks take the flags of golang.org/x/tools/go/analysis/multichecker,
// so a single check runs as, for example,
//
//	cudavet -cargalign ./...
//
// cargalign computes no offsets of its own. Every offset it reports comes
// from [cudakernel.NewSchema], so that kernel parameter alignment has one
// definition and not two.
package main

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/multichecker"
)

var analyzers = []*analysis.Analyzer{
	cargalignAnalyzer,
	unpinnedAnalyzer,
	leakAnalyzer,

	contextThreadAnalyzer,
	unlockDestroyAnalyzer,
	graphShareAnalyzer,
}

func main() {
	multichecker.Main(analyzers...)
}
