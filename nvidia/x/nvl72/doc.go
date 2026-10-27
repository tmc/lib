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
