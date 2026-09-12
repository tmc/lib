// Package nccltest provides Go testing support for NCCL collective correctness
// and performance comparisons with NVIDIA/nccl-tests.
//
// Case.Layout and Case.Bandwidth follow the reference host operation's count
// alignment and reporting conventions. Check uses an independent host oracle;
// it does not infer correctness from NCCL's success status. Sizes builds finite
// message-size sweeps. Run executes integer, float32, and float64 cases on CUDA devices;
// physical multi-GPU equivalence and additional datatypes remain unverified.
package nccltest
