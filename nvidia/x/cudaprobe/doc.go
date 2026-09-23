// Package cudaprobe decodes CUDA API observations and tracks allocation
// lifetimes within one observed process. It does not attach probes, read target
// memory, parse ELF files, or infer tensor shapes from device pointers.
//
// Captures use the format documented by Decode. The collector must pair entry
// arguments with the corresponding return value and record the library build
// identity. Missing or unsupported capture data remains undecoded. Host fixture
// tests do not qualify a collector or a CUDA library on a live target.
package cudaprobe
