package cudapdl_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func TestReviewRegisterWarpGranularity(t *testing.T) {
	driverLoaded, compilerLoaded := driver.Loaded(), nvrtc.Loaded()
	defer func() {
		if driver.Loaded() != driverLoaded || nvrtc.Loaded() != compilerLoaded {
			t.Fatal("register warp granularity tests loaded a native library")
		}
	}()

	t.Run("positive_exact_capacity", func(t *testing.T) {
		prod := testProducerDesc("prod_exact_64")
		prod.ThreadsPerBlock = 512
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 64,
			PeakLiveGPR:        64,
			RegisterQuantum:    256,
		}

		cons := testConsumerDesc("cons_exact_64")
		cons.ThreadsPerBlock = 512
		cons.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 64,
			PeakLiveGPR:        64,
			RegisterQuantum:    256,
		}
		cons.Effects.RequireSingleSMCoResidency = true

		el := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic || el.FallbackReason != "" {
			t.Fatalf("expected eligible exact-capacity pair, got %+v", el)
		}
		if !el.Resources.SingleSMCoResidentPossible {
			t.Fatalf("SingleSMCoResidentPossible = false, want true; resources=%+v", el.Resources)
		}
		if el.Resources.ProducerCTAWarps != 16 || el.Resources.ConsumerCTAWarps != 16 || el.Resources.CombinedCTAWarps != 32 {
			t.Fatalf("unexpected CTA warps: %+v", el.Resources)
		}
		if el.Resources.ProducerCTARegisters != 32768 || el.Resources.ConsumerCTARegisters != 32768 || el.Resources.CombinedCTARegisters != 65536 {
			t.Fatalf("unexpected CTA registers: prod=%d cons=%d combined=%d", el.Resources.ProducerCTARegisters, el.Resources.ConsumerCTARegisters, el.Resources.CombinedCTARegisters)
		}
		if el.Provenance != cudapdl.ProvenanceOwnerDeclared || !el.NoOverlapStillCorrect || el.OverlapGuaranteed {
			t.Fatalf("unexpected provenance/overlap flags: %+v", el)
		}
		if got := cudapdl.NewPlanAdapter("sm_121a").Evaluate(prod, cons); got != el {
			t.Fatalf("PlanAdapter.Evaluate = %+v, want %+v", got, el)
		}
	})

	t.Run("positive_rounded_up_warp_boundary", func(t *testing.T) {
		// Producer: 16 warps, 57 regs/thread -> 32*57=1824 -> rounds to 2048/warp -> 16*2048 = 32768.
		// Consumer: 16 warps, 63 regs/thread -> 32*63=2016 -> rounds to 2048/warp -> 16*2048 = 32768.
		// Combined: 65536 <= 65536.
		prod := testProducerDesc("prod_57_regs")
		prod.ThreadsPerBlock = 512
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 57,
			PeakLiveGPR:        57,
			RegisterQuantum:    256,
		}

		cons := testConsumerDesc("cons_63_regs")
		cons.ThreadsPerBlock = 512
		cons.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 63,
			PeakLiveGPR:        63,
			RegisterQuantum:    256,
		}
		cons.Effects.RequireSingleSMCoResidency = true

		el := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic || el.FallbackReason != "" {
			t.Fatalf("expected eligible warp-rounded pair within 65536 limit, got %+v", el)
		}
		if !el.Resources.SingleSMCoResidentPossible {
			t.Fatalf("SingleSMCoResidentPossible = false, want true; resources=%+v", el.Resources)
		}
		if el.Resources.ProducerCTARegisters != 32768 || el.Resources.ConsumerCTARegisters != 32768 || el.Resources.CombinedCTARegisters != 65536 {
			t.Fatalf("unexpected per-warp rounded CTA registers: prod=%d cons=%d combined=%d", el.Resources.ProducerCTARegisters, el.Resources.ConsumerCTARegisters, el.Resources.CombinedCTARegisters)
		}
		if el.Provenance != cudapdl.ProvenanceOwnerDeclared || !el.NoOverlapStillCorrect || el.OverlapGuaranteed {
			t.Fatalf("unexpected provenance/overlap flags: %+v", el)
		}
		if got := cudapdl.NewPlanAdapter("sm_121a").Evaluate(prod, cons); got != el {
			t.Fatalf("PlanAdapter.Evaluate = %+v, want %+v", got, el)
		}
	})

	t.Run("negative_per_warp_over_capacity", func(t *testing.T) {
		// Exact Root review counterexample:
		// Producer: 544 threads (17 warps), 61 regs -> per-warp ((32*61+255)/256)*256 = 2048 -> 17*2048 = 34816 (whole-CTA would be 33280).
		// Consumer: 512 threads (16 warps), 63 regs -> per-warp ((32*63+255)/256)*256 = 2048 -> 16*2048 = 32768 (whole-CTA would be 32256).
		// Combined: 34816 + 32768 = 67584 > 65536 (whole-CTA combined would be 65536).
		prod := testProducerDesc("prod_17w_61r")
		prod.ThreadsPerBlock = 544
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 61,
			PeakLiveGPR:        61,
			RegisterQuantum:    256,
		}

		cons := testConsumerDesc("cons_16w_63r")
		cons.ThreadsPerBlock = 512
		cons.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 63,
			PeakLiveGPR:        63,
			RegisterQuantum:    256,
		}
		cons.Effects.RequireSingleSMCoResidency = true

		el := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback {
			t.Fatalf("expected fallback when per-warp register allocation (67584) exceeds 65536, got %+v", el)
		}
		if el.Resources.SingleSMCoResidentPossible {
			t.Fatalf("SingleSMCoResidentPossible = true, want false; resources=%+v", el.Resources)
		}
		if el.Resources.ProducerCTARegisters != 34816 || el.Resources.ConsumerCTARegisters != 32768 || el.Resources.CombinedCTARegisters != 67584 {
			t.Fatalf("unexpected per-warp CTA registers: prod=%d (want 34816) cons=%d (want 32768) combined=%d (want 67584)",
				el.Resources.ProducerCTARegisters, el.Resources.ConsumerCTARegisters, el.Resources.CombinedCTARegisters)
		}
		if el.Provenance != cudapdl.ProvenanceNone || el.OverlapGuaranteed {
			t.Fatalf("unexpected fallback provenance/overlap flags: %+v", el)
		}
		if !strings.Contains(el.FallbackReason, "regs=67584/65536") || !strings.Contains(el.FallbackReason, "exceed single-SM co-residency limits on sm_121a") {
			t.Fatalf("FallbackReason = %q, want regs=67584/65536 and exceed single-SM co-residency limits on sm_121a", el.FallbackReason)
		}
		if got := cudapdl.NewPlanAdapter("sm_121a").Evaluate(prod, cons); got != el {
			t.Fatalf("PlanAdapter.Evaluate = %+v, want %+v", got, el)
		}
	})

	t.Run("partial_last_warp_rounding", func(t *testing.T) {
		// Producer: 513 threads -> (513+31)/32 = 17 warps, 61 regs -> 17 * 2048 = 34816.
		// Consumer: 512 threads -> 16 warps, 63 regs -> 16 * 2048 = 32768.
		// Combined: 67584 > 65536.
		prod := testProducerDesc("prod_513t_61r")
		prod.ThreadsPerBlock = 513
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 61,
			PeakLiveGPR:        61,
			RegisterQuantum:    256,
		}

		cons := testConsumerDesc("cons_512t_63r")
		cons.ThreadsPerBlock = 512
		cons.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 63,
			PeakLiveGPR:        63,
			RegisterQuantum:    256,
		}
		cons.Effects.RequireSingleSMCoResidency = true

		elReq := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if elReq.Eligible || elReq.Policy != cudapdl.PolicyStandardFallback || elReq.Resources.SingleSMCoResidentPossible {
			t.Fatalf("expected fallback when RequireSingleSMCoResidency=true with partial 17th warp, got %+v", elReq)
		}
		if elReq.Resources.ProducerCTAWarps != 17 || elReq.Resources.ProducerCTARegisters != 34816 || elReq.Resources.CombinedCTARegisters != 67584 {
			t.Fatalf("unexpected partial-warp resource accounting: %+v", elReq.Resources)
		}

		cons.Effects.RequireSingleSMCoResidency = false
		elOpt := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if !elOpt.Eligible || elOpt.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("expected eligible pair when RequireSingleSMCoResidency=false, got %+v", elOpt)
		}
		if elOpt.Resources.SingleSMCoResidentPossible {
			t.Fatalf("SingleSMCoResidentPossible = true, want false when combined registers (67584) exceed 65536")
		}
		if !elOpt.NoOverlapStillCorrect || elOpt.OverlapGuaranteed || elOpt.Provenance != cudapdl.ProvenanceOwnerDeclared {
			t.Fatalf("unexpected flags on non-co-resident admitted pair: %+v", elOpt)
		}
	})

	t.Run("negative_single_cta_per_warp_exceeds_sm_limit", func(t *testing.T) {
		// Producer: 928 threads (29 warps), 65 regs/thread.
		// Whole-CTA rawRegs = 29 * 32 * 65 = 60320 -> whole-CTA rounded = 60416 <= 65536.
		// Per-warp: ((32 * 65 + 255) / 256) * 256 = 2304 regs/warp -> 29 * 2304 = 66816 > 65536.
		prod := testProducerDesc("prod_29w_65r")
		prod.ThreadsPerBlock = 928
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 65,
			PeakLiveGPR:        65,
			RegisterQuantum:    256,
		}
		cons := testConsumerDesc("cons_ok")

		el := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback {
			t.Fatalf("expected single-CTA per-warp register overflow (66816 > 65536) to be rejected, got %+v", el)
		}
		if el.Provenance != cudapdl.ProvenanceNone || el.OverlapGuaranteed {
			t.Fatalf("unexpected fallback provenance/overlap flags: %+v", el)
		}
		if !strings.Contains(el.FallbackReason, "CTA registers (66816) exceed per-SM register file limit 65536") {
			t.Fatalf("FallbackReason = %q, want substring %q", el.FallbackReason, "CTA registers (66816) exceed per-SM register file limit 65536")
		}
		if got := cudapdl.NewPlanAdapter("sm_121a").Evaluate(prod, cons); got != el {
			t.Fatalf("PlanAdapter.Evaluate = %+v, want %+v", got, el)
		}
	})
}
