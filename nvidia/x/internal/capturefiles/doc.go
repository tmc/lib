// Package capturefiles reads the files a CUDA capture carries beside its
// profile: the PTX and SASS listings, register pressure, tileiras remarks and
// the capture status.
//
// It holds the lookups that cudaprof's subcommands share with the analysis
// packages under x/cudaprof, so that each reads a capture's files the same
// way.
package capturefiles
