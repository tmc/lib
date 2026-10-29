// Package nvl72 emits multi-GPU NVLink 5 and NVLink 6 symmetric-memory
// collective kernels, 72-rank and 144-rank Wide Expert Parallelism (Wide-EP)
// pipelines, O(1) NVSwitch multicast barriers, fabric tensor descriptors, and
// 4-tier GB200/GB300 NVL72 and Vera Rubin NVL144/NVL576 rack and pod planners.
//
// Kernels emitted by this package compile for Hopper (sm_90a), Blackwell
// (sm_100a, sm_103a), and Vera Rubin (sm_107a, sm_107f) with nvcc or NVRTC and
// run on the CPU under the cudahostemu multi-rank host emulator:
//
//   - [AllReduceResidualRMSNorm] fuses in-switch multimem.ld_reduce across
//     ranks with residual addition and RMSNorm in a single kernel launch.
//   - [NVLinkMoECombine] gathers routed expert outputs directly from peer
//     NVLink symmetric memory and applies router-weighted Top-K combination.
//   - [NVL72WideEP] bundles the 4-stage 72-GPU / 18-tray Wide-EP pipeline
//     ([NVL72WideEPPlan], [NVL72WideEPDispatch], [NVL72WideEPExpertScale],
//     [NVL72WideEPCombine]) synchronized via O(1) NVSwitch multicast barriers
//     (multimem.red.release.sys.global.add.u32 + ld.acquire.sys.global.u32)
//     with Grace LPDDR5X NVLink-C2C cold-expert fallback.
//   - [EstimateNVL72WideEP] models single-rack 3-tier traffic and Two-Batch
//     Overlap (TBO) across HBM3e (8.0 TB/s), NVLink 5 (900 GB/s unidirectional
//     per GPU), and Grace NVLink-C2C (450 GB/s per GPU).
//   - [EstimateMultiRackPodEP] models the 4-tier GB200 (192 GB HBM3e,
//     ConnectX-7 400G = 50 GB/s) and GB300 (288 GB 12-Hi HBM3e, ConnectX-8
//     800G PCIe Gen6 = 100 GB/s) multi-rack pod hierarchy with rack-deduped
//     two-stage All-to-All dispatch and Two-Batch Overlap.
//   - [PlanRubinNVL144] and [RubinNVL144Combine] model and emit Vera Rubin
//     VR200 NVL144 single-rack (144 GPUs, NVLink 6 at 1800 GB/s unidirectional
//     per GPU, HBM4 at 13.0 TB/s, Vera CPU LPDDR5X C2C at 900 GB/s) and NVL576
//     4-rack Spectrum-X800 1600G RDMA scale-out pipelines with PTX ISA 9.4
//     `multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2` collectives.
//   - [NewFabricTensor] and [FabricMultimemKernel] provide 64-bit peer-mapped
//     fabric tensor views (`cudaMemAllocationHandleTypeFabric`) across NVL72
//     and NVL144 domains.
//   - [NVL72KVPageTransfer] streams paged KV or compressed MLA latent cache
//     pages from prefill ranks to decode ranks over NVLink peer pointers with
//     system-scope release/acquire ordering and optional C2C staging.
//   - [PlanPDDisaggregation] selects optimal prefill/decode rack and GPU splits
//     under TTFT and TPOT SLAs across GB200, GB300, and Vera Rubin pods.
//   - [ComputeRoutingPlan] routes MoE tokens (DeepSeek-V3 sigmoid with
//     selection-only bias, softmax, or Sinkhorn-Knopp), packs TMA .gather4
//     row quads, and mirrors hot experts into shadow slots;
//     [FusedMoERouter] emits the single-kernel router, checked against it
//     under cudahostemu.
//   - [PlanContextParallel] schedules striped (zig-zag) ring, Ulysses, and
//     2D hybrid context-parallel causal attention with equal causal tile
//     counts per rank and step; [SimulateContextParallel] and
//     [MergeOnlineSoftmaxTiles] check it against monolithic attention, and
//     [ContextParallelRingKernel] runs the ring on emulated ranks.
//   - [AlgebraicCollectiveRules] and [ExtractOptimalDistSchedule] fuse
//     collectives into adjacent GEMM and RMSNorm ops in an e-graph and pick
//     the pulled multimem.ld_reduce form or the pushed per-tile switch
//     reduction by batch size under [DistCostModel]; [EvalDistExpr] checks
//     each rewrite numerically and [MultimemRMSNormKernel] emits the pull.
//   - [ComputeMoEBackwardCombine] computes the MoE backward combine and the
//     gate-logit gradient through normalized sigmoid gates, with traffic
//     for home-rank versus in-fabric reduction;
//     [EmitSwitchMoEBackwardEpiloguePTX] and [SwitchMoEBackwardKernel] emit
//     the pre-scaled expert epilogue.
//   - [DecideColdMoEPlacement] models running a cold expert in place on
//     Grace or Vera cores versus copying it to the GPU over C2C;
//     [EmitC2CSysDoorbellPTX], [AuditC2CCoherence] and [C2CDoorbellKernel]
//     emit and check a .sys-scope release/acquire doorbell between a
//     running kernel and a CPU worker.
//   - [EstimateTPDecode] models a tensor-parallel dense LLaMA decode step on
//     an [NVLinkFabric] ([GB300NVLink5], [GB10]): each rank's weight and KV
//     streaming plus 2·layers+1 all-reduces priced by
//     [NVLinkFabric.AllReduceUs] (one-shot, two-shot, ring or NVLS, with
//     hop latency and link bandwidth), kernel gaps, and the effect of fusing
//     the all-reduce into the GEMM epilogue. Its outputs are modelled; only
//     the GB10 memory term is calibrated against a measured run.
//
// The native gate (CUDA_NVL72_NATIVE=1 under an inherited GPU lock, target
// CUDA_NVL72_NATIVE_ARCH, default sm_121a) compiles emitted kernels with nvcc
// and runs the single-GPU ones; multi-rank paths run only on cudahostemu.
//
// Basic usage:
//
//	est, err := nvl72.EstimateMultiRackPodEP(nvl72.GenGB300, 4, 64, 4096, 8, 288)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Println(est.TotalGPUs, est.Tier4RDMAUniGBps)
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package nvl72
