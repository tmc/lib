// Package main implements pprof-report, a tool that reads cudaprof / pprof
// profiles (including embedded cudacapture archives) and produces a wide,
// token-efficient markdown report across all sample types and stall categories.
//
// Usage:
//
//	pprof-report [flags] <capture.pprof>
//
// Flags:
//
//	-top int
//	    Number of top kernels and hotspot locations to display (default 10).
//	-json
//	    Output structured JSON instead of markdown tables.
package main
