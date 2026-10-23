package cudapdl_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func TestPDLSharedBlockLimit(t *testing.T) {
	driverLoaded, compilerLoaded := driver.Loaded(), nvrtc.Loaded()
	t.Cleanup(func() {
		if driver.Loaded() != driverLoaded || nvrtc.Loaded() != compilerLoaded {
			t.Error("shared block limit controls loaded a native library")
		}
	})
	for _, role := range []string{"producer", "consumer"} {
		for _, tc := range []struct {
			name            string
			static, dynamic uint32
			wantReason      string
		}{
			{"combined exact", 4096, 97280, ""},
			{"combined over", 4096, 97281, "per-block opt-in limit (101376 B)"},
			{"static exact", 101376, 0, ""},
			{"static over", 101377, 0, "per-block opt-in limit (101376 B)"},
			{"dynamic exact", 0, 101376, ""},
			{"dynamic over", 0, 101377, "opt-in limit"},
			{"overflow", math.MaxUint32, 1, "overflows uint32"},
		} {
			t.Run(role+"/"+tc.name, func(t *testing.T) {
				prod, cons := testProducerDesc("producer"), testConsumerDesc("consumer")
				desc := &prod
				if role == "consumer" {
					desc = &cons
				}
				desc.Resources.StaticSharedMemory = tc.static
				desc.Resources.DynamicSharedMemory = tc.dynamic
				got := cudapdl.CheckEligibility("sm_121a", prod, cons)
				if tc.wantReason != "" {
					if got.Eligible || got.Policy != cudapdl.PolicyStandardFallback || got.Provenance != cudapdl.ProvenanceNone || !strings.Contains(got.FallbackReason, role+" kernel") || !strings.Contains(got.FallbackReason, tc.wantReason) {
						t.Fatalf("CheckEligibility = %+v, want %s refusal containing %q", got, role, tc.wantReason)
					}
					return
				}
				if !got.Eligible || got.Policy != cudapdl.PolicyProgrammatic || got.FallbackReason != "" || got.Provenance != cudapdl.ProvenanceOwnerDeclared || !got.NoOverlapStillCorrect || got.OverlapGuaranteed {
					t.Fatalf("CheckEligibility = %+v, want owner-declared admission", got)
				}
				if got.Resources.ProducerSharedBytes+got.Resources.ConsumerSharedBytes != got.Resources.CombinedSharedBytes || got.Resources.CombinedSharedBytes != tc.static+tc.dynamic+4096 {
					t.Fatalf("shared summary = %+v", got.Resources)
				}
				// Individually valid CTAs need not fit on one SM when that is optional.
				if got.Resources.SingleSMCoResidentPossible {
					t.Fatalf("unexpected single-SM possibility: %+v", got.Resources)
				}
				if adapter := cudapdl.NewPlanAdapter("sm_121a").Evaluate(prod, cons); adapter != got {
					t.Fatalf("adapter = %+v, want %+v", adapter, got)
				}
			})
		}
	}
}

func TestPDLSharedSMBudget(t *testing.T) {
	for _, tc := range []struct {
		name                                string
		consumer                            uint32
		require, wantEligible, wantResident bool
	}{
		{"exact required", 51200, true, true, true},
		{"over optional", 51201, false, true, false},
		{"over required", 51201, true, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prod, cons := testProducerDesc("producer"), testConsumerDesc("consumer")
			prod.Resources.StaticSharedMemory = 51200
			cons.Resources.StaticSharedMemory = tc.consumer
			cons.Effects.RequireSingleSMCoResidency = tc.require
			got := cudapdl.CheckEligibility("sm_121a", prod, cons)
			if got.Eligible != tc.wantEligible || got.Resources.SingleSMCoResidentPossible != tc.wantResident || got.Resources.CombinedSharedBytes != 51200+tc.consumer {
				t.Fatalf("CheckEligibility = %+v", got)
			}
			if !tc.wantEligible && (got.Policy != cudapdl.PolicyStandardFallback || !strings.Contains(got.FallbackReason, "single-SM co-residency")) {
				t.Fatalf("combined SM refusal = %+v", got)
			}
		})
	}
}

func TestPDLSharedBlockTargetAndEffects(t *testing.T) {
	prod, cons := testProducerDesc("producer"), testConsumerDesc("consumer")
	for _, target := range []string{"invalid", "sm_95"} {
		if got := cudapdl.CheckEligibility(target, prod, cons); got.Eligible || got.Policy != cudapdl.PolicyStandardFallback || got.FallbackReason == "" {
			t.Fatalf("unknown target %q = %+v", target, got)
		}
	}
	// Use each recorded target's actual per-block cap, including a different SM.
	for _, target := range []string{"sm_90", "sm_100a", "sm_121a"} {
		info, err := cudakernel.ProbedTargetSM(target)
		if err != nil {
			t.Fatal(err)
		}
		prod.Resources.StaticSharedMemory = 4096
		prod.Resources.DynamicSharedMemory = info.MaxOptInSharedBytes - 4096
		if got := cudapdl.CheckEligibility(target, prod, cons); !got.Eligible {
			t.Fatalf("exact %s = %+v", target, got)
		}
		prod.Resources.DynamicSharedMemory++
		if got := cudapdl.CheckEligibility(target, prod, cons); got.Eligible || !strings.Contains(got.FallbackReason, "per-block opt-in limit") {
			t.Fatalf("over %s = %+v", target, got)
		}
	}
	prod = testProducerDesc("producer")
	cons.Effects.WaitPlacement = cudapdl.WaitPlacementUnknown
	if got := cudapdl.CheckEligibility("sm_121a", prod, cons); got.Eligible || !strings.Contains(got.FallbackReason, "wait placement is unknown") {
		t.Fatalf("invalid effects = %+v", got)
	}
}
