// Package cudaprobe decodes CUDA API observations and tracks allocation
// lifetimes within one observed process. It does not attach probes, read target
// memory, parse ELF files, or infer tensor shapes from device pointers.
//
// Supported linux-amd64 SysV and linux-arm64 AAPCS64 captures include launch geometry, opaque graph
// executable submissions, linear copy arguments, synchronization scopes, and
// synchronous and stream-ordered allocation/free calls, and explicit context
// and stream controls. Stream-ordered observations never change Tracker's
// synchronous allocation lifetimes. Context handles never become inferred
// allocation address-space identities. Runtime copies preserve cudaMemcpyKind;
// Driver copies preserve the direction named by the API. API observations do
// not measure device duration, expand graph nodes, or establish actual transfer
// direction for cudaMemcpyDefault. Peer, pitched, multidimensional, legacy
// Driver copy signatures, and per-thread-default-stream suffixes are refused.
//
// Signature references are the NVIDIA CUDA Runtime and Driver API manuals:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html and
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html.
// Stream-ordered memory semantics are documented at
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html.
//
// Captures use the format documented by Decode. The collector must pair entry
// arguments with the corresponding return value and record the library build
// identity. Missing or unsupported capture data remains undecoded. Host fixture
// tests do not qualify a collector or a CUDA library on a live target.
package cudaprobe
