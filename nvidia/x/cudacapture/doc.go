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
//
// Beyond the container, the package builds profile frames from retained code
// and checks that the artifacts describe the code that ran.
//
// FrameBuilder turns a Code and Kernel identity into [cudaprof] frames for
// SASS instructions, PTX lines, and source lines. Addresses come from
// PTXAddresses, parsed from the retained PTX; an unmapped program counter
// stays unknown rather than being attributed to a nearby line.
//
// SASS records the paths its producer compiled from, which are rarely the
// paths a reader holds. SourceMapByBase reads those paths back out of the
// disassembly and pairs them with named artifacts, reporting an error when two
// sources share a basename. RewriteSASSSources applies the map and retains
// every path it does not cover.
//
// A module compiled from several files is one translation unit, which is what
// a single NVRTC compilation of a file that includes others produces.
// CodeFiles.Sources names every CUDA file that took part; LoadCode resolves
// each kernel's definition against whichever of them defines it, and
// Kernel.SourceFile reports which. The line-table check runs per file and only
// over the files the capture supplies: a file the disassembly names but the
// capture does not bundle has no line table to check against, so its lines are
// unknowable rather than unclaimed.
//
// VerifyModule resolves a module generation against a required image digest
// recomputed over the retained CUBIN. The digest is a producer record checked
// against retained bytes, not a driver attestation, and a kernel name that
// resolves is not module-generation proof.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudacapture
