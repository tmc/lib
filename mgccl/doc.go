// Package mgccl coordinates higher-level collective communication flows.
//
// Package mgccl does not bind a C library. It orchestrates generated local
// collective packages such as gopccl and leaves NCCL, RCCL, PCCL, CUDA, and
// RDMA datapath checks behind their package-specific manual gates.
package mgccl
