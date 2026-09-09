// Package cudacapture reads and writes portable CUDA execution captures
// containing pprof profiles, source code, compilation artifacts, and disassembly.
//
// A capture bundles an execution profile with the exact source, PTX, CUBIN,
// and SASS disassembly files corresponding to the measured kernels. Captures
// can be stored unpacked in a directory or serialized as a single self-contained
// pprof file with embedded artifacts.
package cudacapture
