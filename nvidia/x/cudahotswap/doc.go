// Package cudahotswap reports kernel candidate outcomes and compares outputs.
// Device faults and timeouts require the owning process to stop using its CUDA
// context. A successful host comparison does not establish native execution.
package cudahotswap
