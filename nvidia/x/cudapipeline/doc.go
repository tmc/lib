// Package cudapipeline provides static verification and transaction-barrier (mbarrier)
// analysis for asynchronous memory pipelines targeting Hopper and Blackwell GPUs.
//
// The package models multi-stage producer-consumer pipelines built with NVIDIA Tensor Memory
// Accelerator (TMA) bulk tensor copies (cp.async.bulk.tensor.2d) and PTX transaction barriers
// (mbarrier.arrive.expect_tx).
//
// # Architecture Support and sm_121a Hardware Realities
//
// Modern NVIDIA architectures differ significantly in their hardware instruction support:
//   - Hopper (sm_90): supports TMA, mbarrier, and warp-group matrix-multiply-accumulate (wgmma).
//   - Blackwell (sm_100): supports TMA, mbarrier, multimem, and tensor-core generation 5 (tcgen05).
//   - Blackwell (sm_121a): accepts TMA, mbarrier, and multimem, but rejects wgmma and tcgen05.
//
// Pipelines targeting sm_121a must rely on standard tensor core instructions and avoid
// wgmma or tcgen05 invocations. This package statically verifies instruction and alignment
// constraints for the chosen target architecture.
//
// # Verification Checks
//
// The [Validator] performs static analysis on pipeline configurations to detect common bugs
// before kernel compilation or GPU execution:
//   - Transaction Count Mismatch: verifies that mbarrier expected transaction bytes match the
//     exact TMA box payload bytes, preventing permanent hangs (under-counting) or race conditions
//     (over-counting).
//   - Phase Parity Consistency: validates that producer arrival phase parity matches consumer
//     wait phase parity across pipeline iterations, preventing deadlocks.
//   - Multi-Stage Buffer Depth: analyzes memory round-trip latency against stage compute intervals
//     to verify whether double-buffering or triple-buffering completely hides memory latency.
//   - Alignment and Stride Conformance: enforces 16-byte shared memory alignment, 8-byte mbarrier
//     alignment, 16-byte row stride multiples, and 128-byte box width constraints for sm_121a and sm_90.
//
// # Basic Usage
//
//	box := cudapipeline.TMABox{
//		DimX:        64,
//		DimY:        64,
//		ElementSize: 2, // 16-bit elements (e.g. fp16 or bf16)
//	}
//	pipe, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
//		TargetArch:            cudapipeline.ArchSM121a,
//		NumStages:             3, // Triple-buffering
//		Box:                   box,
//		MemoryLatencyCycles:   600,
//		ComputeIntervalCycles: 300,
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	report := pipe.Validate()
//	if !report.Valid() {
//		log.Fatalf("pipeline invalid: %v", report.Errors())
//	}
package cudapipeline
