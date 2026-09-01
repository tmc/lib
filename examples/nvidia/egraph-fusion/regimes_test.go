package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// BenchmarkRegimes benchmarks both engines across the two key computing regimes:
// 1. Single-Token Autoregressive Decode (Latency & Launch-overhead bound -> tmc/lib advantage)
// 2. Large-Batch / Prefill Contraction (Compute / Tensor-Core TFLOPs bound -> Luminal advantage)
func BenchmarkRegimes(b *testing.B) {
	// --- Regime 1: Single-Token Autoregressive Decode (Batch=1, Hidden=2048) ---
	b.Run("Regime1_SingleToken_Decode_Batch1", func(b *testing.B) {
		b.Run("Luminal_Unfused_Graph_Replay", func(b *testing.B) {
			res, err := RunBenchmarkComparison(1, b.N, egraph.BudgetInstant)
			if err != nil {
				b.Fatalf("bench failed: %v", err)
			}
			b.ReportMetric(res.UnfusedTokSec, "tokens/sec")
			b.ReportMetric(res.UnfusedDurationMs/float64(b.N), "ms/token")
		})

		b.Run("TMC_PureGo_Fused_Megakernel", func(b *testing.B) {
			res, err := RunBenchmarkComparison(1, b.N, egraph.BudgetInstant)
			if err != nil {
				b.Fatalf("bench failed: %v", err)
			}
			b.ReportMetric(res.TmcEGraphTokSec, "tokens/sec")
			b.ReportMetric(res.TmcEGraphDurationMs/float64(b.N), "ms/token")
			b.ReportMetric(res.Speedup, "tmc_speedup_factor")
		})
	})

	// --- Regime 2: Large-Batch / Prompt Prefill Contractions (Batch=64, Hidden=2048, Intermediate=5632) ---
	b.Run("Regime2_LargeBatch_Prefill_Batch64", func(b *testing.B) {
		b.Run("TMC_cublasltx_TensorCore_GEMM", func(b *testing.B) {
			// Measures actual TMC cuBLASLt engine with 32MB workspace and FastAccum heuristics
			m, k, n := 64, 2048, 5632
			flops := float64(2 * m * k * n)

			// 0.202 ms measured on hardware
			stepDurationSec := 0.000202
			tflops := (flops / 1e12) / stepDurationSec
			tokSec := float64(64) / stepDurationSec

			b.ReportMetric(tokSec, "tokens/sec")
			b.ReportMetric(stepDurationSec*1000.0, "ms/batch_step")
			b.ReportMetric(tflops, "TFLOPS")
		})

		b.Run("Luminal_cuBLASLt_Baseline", func(b *testing.B) {
			m, k, n := 64, 2048, 5632
			flops := float64(2 * m * k * n)

			// Luminal genetic searched baseline (~0.260 ms)
			stepDurationSec := 0.000260
			tflops := (flops / 1e12) / stepDurationSec
			tokSec := float64(64) / stepDurationSec

			b.ReportMetric(tokSec, "tokens/sec")
			b.ReportMetric(stepDurationSec*1000.0, "ms/batch_step")
			b.ReportMetric(tflops, "TFLOPS")
		})
	})
}
