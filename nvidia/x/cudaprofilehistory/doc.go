// Package cudaprofilehistory retains bounded, immutable observations of CUDA
// artifacts and profiling receipts. Keys bind content and declared provenance;
// they do not authenticate a producer or qualify imported GPU measurements.
//
// Comparisons keep code generation, resources, correctness and performance
// separate. Bisect plans use supplied observations and never execute a kernel.
// Stores are safe for concurrent use. Snapshot replacement and file locking
// belong to the caller; this package does not implement cross-process append.
package cudaprofilehistory
