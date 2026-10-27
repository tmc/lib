package hopper

import "github.com/tmc/lib/nvidia/x/tcgen05"

// TCGen05AdvancedPrelude is the self-contained device preamble emitted for
// advanced sm_100a/sm_103a tcgen05 kernels.
const TCGen05AdvancedPrelude = tcgen05.TCGen05AdvancedPrelude

// TCGen05Cluster2GEMM aliases [tcgen05.TCGen05Cluster2GEMM].
type TCGen05Cluster2GEMM = tcgen05.TCGen05Cluster2GEMM

// TCGen05BlockScaledGEMM aliases [tcgen05.TCGen05BlockScaledGEMM].
type TCGen05BlockScaledGEMM = tcgen05.TCGen05BlockScaledGEMM

// TCGen05TMEMFlashAttention aliases [tcgen05.TCGen05TMEMFlashAttention].
type TCGen05TMEMFlashAttention = tcgen05.TCGen05TMEMFlashAttention

// TCGen05ScaleVec4XGEMM aliases [tcgen05.TCGen05ScaleVec4XGEMM].
type TCGen05ScaleVec4XGEMM = tcgen05.TCGen05ScaleVec4XGEMM
