// Package cudacapture reads and writes portable CUDA execution captures
// containing pprof profiles, source code, compilation artifacts, and disassembly.
//
// A capture bundles an execution profile with code artifacts and their
// provenance. Version 2 recordings require source, PTX, CUBIN, and SASS. The
// version 3 extension records explicit unavailable dispositions, module
// generations, typed metric descriptors, and observations, so a stripped or
// third-party image remains inspectable without fabricated files or PCs. The
// v3 envelope is distinct from v2, and readers retain supported older files.
// Captures can be stored unpacked in a directory or serialized as a single
// self-contained pprof file. Readers enforce bounded file and input sizes.
package cudacapture
