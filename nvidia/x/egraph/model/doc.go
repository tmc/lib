// Package model provides transformer decoder architecture configuration
// parsing, layer execution graph IR construction, roofline cost estimation,
// end-to-end multi-architecture production compilation across Hopper (sm_90a),
// Blackwell datacenter and workstation (sm_100a, sm_103a, sm_120a, sm_121a),
// and Vera Rubin (sm_107a, sm_107f) with PTX ISA 9.4 and NVL72/NVL144/NVL576
// collectives, multi-architecture fatbinary packaging, and safetensors layer
// weight extraction and NVFP4 WGMMA repacking.
//
// Production lowering entry points:
//
//   - [CompileProductionModel] compiles a parsed [ModelConfig] and target SM
//     compute capability (`sm_90a`, `sm_100a`, `sm_103a`, `sm_107a`, `sm_107f`,
//     `sm_120a`, `sm_121a`) into a verified [ProductionModelPlan] containing
//     the architecture dispatch plan ([SelectArchDispatchPlan]), specialized
//     CUDA C++ megakernel translation units, verified TMEM lifecycle traces
//     (`cudatmem.VerifyTrace`), multi-rack NVL72/NVL144/NVL576 collective plans,
//     and multi-architecture fatbinary manifests.
//   - [LowerModelSM100] specializes a [LayerGraph] for Blackwell sm_100a
//     tcgen05/CLC and NVLink TP/EP collective operations.
//   - [BuildFatbin] packages per-target PTX/CUBIN entries across Hopper,
//     Blackwell, and Vera Rubin with automatic runtime selection via
//     [SelectBestTarget].
//   - [StreamLoader.PackTensor] repacks row-major checkpoint weights into
//     tile-major TMA-swizzled BF16, FP8, NVFP4 scale_vec::4X, 2:4 sparse
//     NVFP4 or Rubin 3-bit LUT layouts with matching cudatma descriptors;
//     [UnpackTensorToFP32] inverts them on the host. [MapSafetensors] reads
//     tensors in place from a read-only mapping, [PlanStream] schedules
//     copy/repack overlap on two streams as a host model, and
//     [SwizzleRepackCUDA] emits the device repack kernel.
//   - [PackAOTArtifact] checks a fatbin ([InspectFatbin]) against per-target
//     [ArchLaunchSpec] launch geometry and TMA descriptor templates, and the
//     resulting [AOTArtifact] emits a standalone go:embed package that
//     loads and launches it through nvidia/driver without cgo
//     ([AOTArtifact.EmitGoEmbedPackage]) or a C header
//     ([AOTArtifact.EmitCHeader]).
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package model
