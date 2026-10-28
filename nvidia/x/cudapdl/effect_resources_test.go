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

func mustConstructPTXArtifact(t *testing.T, entry, target string, opts ...cudakernel.ArtifactOption) *cudakernel.Artifact {
	t.Helper()
	raw := []byte(".version 8.0\n.target " + target + "\n.visible .entry " + entry + "()\n{\n\tret;\n}\n")
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, raw, entry, cudakernel.Schema{}, target, opts...)
	if err != nil {
		t.Fatalf("NewArtifact(%s, %s): %v", entry, target, err)
	}
	return art
}

func TestPDLEffectAndPlacementControls(t *testing.T) {
	driverLoaded, compilerLoaded := driver.Loaded(), nvrtc.Loaded()
	defer func() {
		if driver.Loaded() != driverLoaded || nvrtc.Loaded() != compilerLoaded {
			t.Fatal("host effect/resource tests loaded a native library")
		}
	}()

	// 1. Safe hoisted-preamble pair and safe entry-sync pair.
	t.Run("safe hoisted preamble pair", func(t *testing.T) {
		prod := testProducerDesc("prod")
		cons := testConsumerDesc("cons")
		el := cudapdl.CheckEligibility("sm_90", prod, cons)
		if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic || el.FallbackReason != "" {
			t.Fatalf("expected eligible programmatic pair, got %+v", el)
		}
		if el.Provenance != cudapdl.ProvenanceOwnerDeclared {
			t.Fatalf("Provenance = %v, want %v", el.Provenance, cudapdl.ProvenanceOwnerDeclared)
		}
		if !el.ConsumerHasIndependentPreamble {
			t.Fatal("expected ConsumerHasIndependentPreamble=true for hoisted preamble consumer")
		}
		if el.OverlapGuaranteed {
			t.Fatal("OverlapGuaranteed must be false for static/owner-declared planning")
		}
		if !el.NoOverlapStillCorrect {
			t.Fatal("NoOverlapStillCorrect must be true for admitted pair")
		}

		adapter := cudapdl.NewPlanAdapter("sm_90")
		if got := adapter.Evaluate(prod, cons); got != el {
			t.Fatalf("PlanAdapter.Evaluate = %+v, want %+v", got, el)
		}
	})

	t.Run("safe entry-sync consumer without independent preamble", func(t *testing.T) {
		prod := testProducerDesc("prod")
		cons := testConsumerDesc("cons_entry_sync")
		cons.HasIndependentWork = false
		cons.Effects.WaitPlacement = cudapdl.WaitAtEntryBeforeDependentBody
		cons.Effects.BeforeWait = nil

		el := cudapdl.CheckEligibility("sm_100a", prod, cons)
		if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("entry-sync consumer should be eligible: %+v", el)
		}
		if el.ConsumerHasIndependentPreamble {
			t.Fatal("ConsumerHasIndependentPreamble must be false when BeforeWait is empty")
		}
		if el.Provenance != cudapdl.ProvenanceOwnerDeclared || el.OverlapGuaranteed || !el.NoOverlapStillCorrect {
			t.Fatalf("unexpected provenance/overlap flags: %+v", el)
		}
	})

	// 2. Negative controls for unknown/unsafe trigger, wait, preamble, and effect kinds.
	var overMaxEffects []cudapdl.PhaseEffect
	for i := 0; i <= cudapdl.MaxPhaseEffects; i++ {
		overMaxEffects = append(overMaxEffects, cudapdl.PhaseEffect{Kind: cudapdl.EffectPureCompute})
	}

	negativeCases := []struct {
		name       string
		mutate     func(prod, cons *cudapdl.KernelDesc)
		wantReason string
	}{
		{
			name: "producer unknown trigger placement",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.TriggerPlacement = cudapdl.TriggerPlacementUnknown
			},
			wantReason: "trigger placement is unknown or unverified",
		},
		{
			name: "producer trigger before shared writes complete",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.TriggerPlacement = cudapdl.TriggerBeforeSharedWritesComplete
			},
			wantReason: "triggers dependents before shared handoff writes complete",
		},
		{
			name: "producer out-of-range trigger placement",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.TriggerPlacement = cudapdl.TriggerPlacement(99)
			},
			wantReason: "unsupported trigger placement",
		},
		{
			name: "producer empty before_launch_dependents",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.BeforeLaunchDependents = nil
			},
			wantReason: "declares no before_launch_dependents phase effects",
		},
		{
			name: "producer writes no handoff resource before trigger",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.BeforeLaunchDependents = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "input", Access: cudapdl.AccessRead, Bytes: 1024},
				}
			},
			wantReason: "before_launch_dependents phase writes no handoff memory resource",
		},
		{
			name: "producer declares consumer wait phase on producer role",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectPureCompute}}
			},
			wantReason: "declares consumer wait phases on producer role",
		},
		{
			name: "consumer declares producer trigger phase on consumer role",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeLaunchDependents = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectPureCompute}}
			},
			wantReason: "declares producer launch_dependents phases on consumer role",
		},
		{
			name: "consumer unknown wait placement",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.WaitPlacement = cudapdl.WaitPlacementUnknown
			},
			wantReason: "wait placement is unknown or unverified",
		},
		{
			name: "consumer wait after dependent access",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.WaitPlacement = cudapdl.WaitAfterDependentAccess
			},
			wantReason: "wait placement occurs after dependent resource access",
		},
		{
			name: "consumer out-of-range wait placement",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.WaitPlacement = cudapdl.WaitPlacement(99)
			},
			wantReason: "unsupported wait placement",
		},
		{
			name: "dishonest preamble HasIndependentWork true with empty BeforeWait",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.HasIndependentWork = true
				cons.Effects.BeforeWait = nil
			},
			wantReason: "claims HasIndependentWork without declared before_wait preamble effects",
		},
		{
			name: "dishonest preamble HasIndependentWork false with non-empty BeforeWait",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.HasIndependentWork = false
			},
			wantReason: "declares before_wait preamble effects with HasIndependentWork=false",
		},
		{
			name: "WaitAfterIndependentPreamble with empty BeforeWait and HasIndependentWork false",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.HasIndependentWork = false
				cons.Effects.BeforeWait = nil
				cons.Effects.WaitPlacement = cudapdl.WaitAfterIndependentPreamble
			},
			wantReason: "declares WaitAfterIndependentPreamble without independent before_wait preamble work",
		},
		{
			name: "WaitAtEntryBeforeDependentBody with non-empty BeforeWait",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.WaitPlacement = cudapdl.WaitAtEntryBeforeDependentBody
			},
			wantReason: "declares WaitAtEntryBeforeDependentBody with non-empty before_wait preamble",
		},
		{
			name: "consumer empty after_wait",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.AfterWait = nil
			},
			wantReason: "declares no after_wait phase effects",
		},
		{
			name: "exceeding MaxPhaseEffects bound",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = overMaxEffects
			},
			wantReason: "exceeding maximum 64",
		},
		{
			name: "unknown effect kind zero",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectKindUnknown}}
			},
			wantReason: "unknown or unspecified effect kind",
		},
		{
			name: "unsafe EffectVolatileMMIO",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.BeforeLaunchDependents = append(prod.Effects.BeforeLaunchDependents, cudapdl.PhaseEffect{Kind: cudapdl.EffectVolatileMMIO})
			},
			wantReason: "unsafe or unsupported effect EffectVolatileMMIO",
		},
		{
			name: "unsafe EffectHostCallback",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectHostCallback}}
			},
			wantReason: "unsafe or unsupported effect EffectHostCallback",
		},
		{
			name: "unsafe EffectSystemAtomic",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectSystemAtomic}}
			},
			wantReason: "unsafe or unsupported effect EffectSystemAtomic",
		},
		{
			name: "unsafe EffectPeerMemory",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.AfterWait = append(cons.Effects.AfterWait, cudapdl.PhaseEffect{Kind: cudapdl.EffectPeerMemory})
			},
			wantReason: "unsafe or unsupported effect EffectPeerMemory",
		},
		{
			name: "unsafe EffectDynamicParallelism",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectDynamicParallelism}}
			},
			wantReason: "unsafe or unsupported effect EffectDynamicParallelism",
		},
		{
			name: "unsafe EffectOpaqueExternal",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectOpaqueExternal}}
			},
			wantReason: "unsafe or unsupported effect EffectOpaqueExternal",
		},
		{
			name: "unrecognized EffectKind value",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectKind(77)}}
			},
			wantReason: "unrecognized effect kind EffectKind(77)",
		},
		{
			name: "EffectPureCompute with non-empty resource",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectPureCompute, Resource: "scratch"}}
			},
			wantReason: "EffectPureCompute must not specify a memory resource",
		},
		{
			name: "EffectMemory with empty resource",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectMemory, Resource: "   ", Access: cudapdl.AccessRead, Bytes: 64}}
			},
			wantReason: "EffectMemory requires a valid non-empty resource name",
		},
		{
			name: "EffectMemory with NUL in resource name",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectMemory, Resource: "bad\x00name", Access: cudapdl.AccessRead, Bytes: 64}}
			},
			wantReason: "EffectMemory requires a valid non-empty resource name",
		},
		{
			name: "EffectMemory with AccessNone",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectMemory, Resource: "weights", Access: cudapdl.AccessNone, Bytes: 64}}
			},
			wantReason: "invalid memory access mode AccessNone",
		},
		{
			name: "EffectMemory with unrecognized AccessMode",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectMemory, Resource: "weights", Access: cudapdl.AccessMode(88), Bytes: 64}}
			},
			wantReason: "invalid memory access mode AccessMode(88)",
		},
		{
			name: "EffectMemory with zero Bytes",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{Kind: cudapdl.EffectMemory, Resource: "weights", Access: cudapdl.AccessRead, Bytes: 0}}
			},
			wantReason: "requires positive byte count",
		},
		{
			name: "EffectMemory Base+Bytes uint64 overflow",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{{
					Kind:     cudapdl.EffectMemory,
					Resource: "weights",
					Access:   cudapdl.AccessRead,
					Base:     math.MaxUint64 - 127,
					Bytes:    256,
				}}
			},
			wantReason: "overflows uint64",
		},
		{
			name: "duplicate memory resource within same phase",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "weights", Access: cudapdl.AccessRead, Bytes: 512},
					{Kind: cudapdl.EffectMemory, Resource: "weights", Access: cudapdl.AccessRead, Bytes: 512},
				}
			},
			wantReason: "duplicate or overlapping memory effects",
		},
	}

	adapter := cudapdl.NewPlanAdapter("sm_90")
	for _, tc := range negativeCases {
		t.Run(tc.name, func(t *testing.T) {
			prod := testProducerDesc("prod")
			cons := testConsumerDesc("cons")
			tc.mutate(&prod, &cons)

			el := cudapdl.CheckEligibility("sm_90", prod, cons)
			if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback {
				t.Fatalf("expected fallback, got %+v", el)
			}
			if el.Provenance != cudapdl.ProvenanceNone {
				t.Fatalf("Provenance = %v, want ProvenanceNone", el.Provenance)
			}
			if !strings.Contains(el.FallbackReason, tc.wantReason) {
				t.Fatalf("FallbackReason = %q, want substring %q", el.FallbackReason, tc.wantReason)
			}
			if got := adapter.Evaluate(prod, cons); got != el {
				t.Fatalf("PlanAdapter.Evaluate = %+v, want %+v", got, el)
			}
		})
	}
}

func TestPDLCrossPhaseHazardControls(t *testing.T) {
	t.Run("safe shared read-only input between producer and consumer preamble", func(t *testing.T) {
		prod := testProducerDesc("prod")
		cons := testConsumerDesc("cons")
		// Both producer and consumer BeforeWait read immutable "input" buffer.
		cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
			{Kind: cudapdl.EffectMemory, Resource: "input", Access: cudapdl.AccessRead, Base: 0x10000, Bytes: 1024},
		}
		prod.Effects.BeforeLaunchDependents[0].Base = 0x10000

		el := cudapdl.CheckEligibility("sm_90", prod, cons)
		if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("shared read-only input should be eligible: %+v", el)
		}
	})

	hazardCases := []struct {
		name       string
		mutate     func(prod, cons *cudapdl.KernelDesc)
		wantReason string
	}{
		{
			name: "consumer before_wait prematurely reads producer handoff buffer",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "handoff", Access: cudapdl.AccessRead, Bytes: 1024},
				}
			},
			wantReason: `before_wait preamble accesses producer handoff resource "handoff" before griddepcontrol.wait`,
		},
		{
			name: "consumer before_wait prematurely writes producer input buffer",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "input", Access: cudapdl.AccessWrite, Bytes: 1024},
				}
			},
			wantReason: `before_wait preamble mutates shared producer resource "input" before griddepcontrol.wait`,
		},
		{
			name: "consumer before_wait races with producer after_launch_dependents write",
			mutate: func(prod, cons *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "scratch", Access: cudapdl.AccessWrite, Bytes: 1024},
				}
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "scratch", Access: cudapdl.AccessRead, Bytes: 1024},
				}
			},
			wantReason: `concurrent shared-resource hazard between producer kernel "prod" after_launch_dependents and consumer kernel "cons" before_wait on resource "scratch"`,
		},
		{
			name: "consumer before_wait write races with producer after_launch_dependents read",
			mutate: func(prod, cons *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "scratch", Access: cudapdl.AccessRead, Bytes: 1024},
				}
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "scratch", Access: cudapdl.AccessAtomic, Bytes: 1024},
				}
			},
			wantReason: `concurrent shared-resource hazard between producer kernel "prod" after_launch_dependents and consumer kernel "cons" before_wait on resource "scratch"`,
		},
		{
			name: "consumer before_wait overlaps producer handoff by concrete address range under different name",
			mutate: func(prod, cons *cudapdl.KernelDesc) {
				prod.Effects.BeforeLaunchDependents[1].Base = 0x20000
				prod.Effects.BeforeLaunchDependents[1].Bytes = 4096
				cons.Effects.AfterWait[0].Base = 0x20000
				cons.Effects.AfterWait[0].Bytes = 4096
				cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "aliased_slice", Access: cudapdl.AccessRead, Base: 0x20800, Bytes: 512},
				}
			},
			wantReason: `before_wait preamble accesses producer handoff resource "aliased_slice" before griddepcontrol.wait`,
		},
		{
			name: "inconsistent address range for same logical resource name",
			mutate: func(prod, cons *cudapdl.KernelDesc) {
				prod.Effects.BeforeLaunchDependents[1].Base = 0x20000
				cons.Effects.AfterWait[0].Base = 0x30000
			},
			wantReason: `inconsistent address range for resource "handoff"`,
		},
		{
			name: "consumer after_wait does not read any producer handoff output",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Effects.AfterWait = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "unrelated_buf", Access: cudapdl.AccessRead, Bytes: 1024},
					{Kind: cudapdl.EffectMemory, Resource: "output", Access: cudapdl.AccessWrite, Bytes: 1024},
				}
			},
			wantReason: `after_wait phase does not read any handoff resource written by producer kernel "prod" before_launch_dependents`,
		},
		{
			name: "producer after_launch_dependents mutates handoff buffer after trigger",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "handoff", Access: cudapdl.AccessWrite, Bytes: 1024},
				}
			},
			wantReason: `after_launch_dependents trailing phase mutates handoff resource "handoff" after trigger`,
		},
		{
			name: "producer after_launch_dependents writes consumer after_wait output",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "output", Access: cudapdl.AccessReadWrite, Bytes: 1024},
				}
			},
			wantReason: `after_launch_dependents trailing phase accesses handoff/consumer resource "output" after trigger`,
		},
		{
			name: "producer after_launch_dependents reads consumer after_wait output",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{
					{Kind: cudapdl.EffectMemory, Resource: "output", Access: cudapdl.AccessRead, Bytes: 1024},
				}
			},
			wantReason: `after_launch_dependents trailing phase accesses handoff/consumer resource "output" after trigger`,
		},
	}

	for _, tc := range hazardCases {
		t.Run(tc.name, func(t *testing.T) {
			prod := testProducerDesc("prod")
			cons := testConsumerDesc("cons")
			tc.mutate(&prod, &cons)

			el := cudapdl.CheckEligibility("sm_90", prod, cons)
			if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback {
				t.Fatalf("expected hazard fallback, got %+v", el)
			}
			if !strings.Contains(el.FallbackReason, tc.wantReason) {
				t.Fatalf("FallbackReason = %q, want substring %q", el.FallbackReason, tc.wantReason)
			}
		})
	}
}

func TestPDLResourceAndTargetLimits(t *testing.T) {
	t.Run("artifact supplies resources and JIT ThreadsPerBlock when KernelDesc fields unset", func(t *testing.T) {
		res := cudakernel.ResourceRequirements{
			StaticSharedMemory: 8192,
			AllocatedRegisters: 48,
			PeakLiveGPR:        40,
			RegisterQuantum:    256,
		}
		jit := cudakernel.JITOptions{ThreadsPerBlock: 256}
		prodArt := mustConstructPTXArtifact(t, "prod", "sm_90", cudakernel.WithResources(res), cudakernel.WithJIT(jit))
		consArt := mustConstructPTXArtifact(t, "cons", "sm_90", cudakernel.WithResources(res), cudakernel.WithJIT(jit))

		prod := testProducerDesc("prod")
		prod.Artifact = prodArt
		prod.Resources = nil
		prod.ThreadsPerBlock = 0

		cons := testConsumerDesc("cons")
		cons.Artifact = consArt
		cons.Resources = nil
		cons.ThreadsPerBlock = 0

		el := cudapdl.CheckEligibility("sm_90", prod, cons)
		if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("expected artifact-backed resources to be eligible: %+v", el)
		}
		if el.Resources.ProducerCTAWarps != 8 || el.Resources.ConsumerCTAWarps != 8 || el.Resources.CombinedCTAWarps != 16 {
			t.Fatalf("unexpected warp summary: %+v", el.Resources)
		}
		if el.Resources.ProducerCTARegisters != 256*48 || el.Resources.CombinedCTARegisters != 2*256*48 {
			t.Fatalf("unexpected register summary: %+v", el.Resources)
		}
		if !el.Resources.SingleSMCoResidentPossible {
			t.Fatalf("expected SingleSMCoResidentPossible=true, got %+v", el.Resources)
		}
	})

	t.Run("conflict between KernelDesc.Resources and Artifact.Resources refused", func(t *testing.T) {
		artRes := cudakernel.ResourceRequirements{StaticSharedMemory: 4096, AllocatedRegisters: 32, PeakLiveGPR: 24}
		prodArt := mustConstructPTXArtifact(t, "prod", "sm_90", cudakernel.WithResources(artRes))

		prod := testProducerDesc("prod")
		prod.Artifact = prodArt
		prod.Resources = &cudakernel.ResourceRequirements{StaticSharedMemory: 8192, AllocatedRegisters: 64, PeakLiveGPR: 24}
		cons := testConsumerDesc("cons")

		el := cudapdl.CheckEligibility("sm_90", prod, cons)
		if el.Eligible || !strings.Contains(el.FallbackReason, "resource requirements conflict with artifact resources") {
			t.Fatalf("expected artifact resource conflict fallback, got %+v", el)
		}
	})

	t.Run("conflict between KernelDesc.ThreadsPerBlock and Artifact.JITOptions.ThreadsPerBlock refused", func(t *testing.T) {
		prodArt := mustConstructPTXArtifact(t, "prod", "sm_90", cudakernel.WithJIT(cudakernel.JITOptions{ThreadsPerBlock: 256}))

		prod := testProducerDesc("prod")
		prod.Artifact = prodArt
		prod.ThreadsPerBlock = 128
		cons := testConsumerDesc("cons")

		el := cudapdl.CheckEligibility("sm_90", prod, cons)
		if el.Eligible || !strings.Contains(el.FallbackReason, "ThreadsPerBlock (128) conflicts with artifact JITOptions.ThreadsPerBlock (256)") {
			t.Fatalf("expected ThreadsPerBlock conflict fallback, got %+v", el)
		}
	})

	t.Run("unprobed estimated target geometry sm_95 refused", func(t *testing.T) {
		prod := testProducerDesc("prod")
		cons := testConsumerDesc("cons")
		el := cudapdl.CheckEligibility("sm_95", prod, cons)
		if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback || !strings.Contains(el.FallbackReason, "unprobed estimated resource limits") {
			t.Fatalf("expected unprobed target fallback on sm_95, got %+v", el)
		}
	})

	t.Run("shared memory target distinction sm_121a vs sm_100a", func(t *testing.T) {
		prod := testProducerDesc("prod")
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory:  32768,
			DynamicSharedMemory: 87232, // total 120,000 B (> 102,400 B on sm_121a, <= 233,472 B on sm_100a)
			AllocatedRegisters:  32,
		}
		cons := testConsumerDesc("cons")

		el121 := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if el121.Eligible || !strings.Contains(el121.FallbackReason, "exceeds target sm_121a SM limit (102400 B)") {
			t.Fatalf("expected sm_121a shared memory rejection, got %+v", el121)
		}

		el100 := cudapdl.CheckEligibility("sm_100a", prod, cons)
		if !el100.Eligible || el100.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("expected sm_100a to admit 120 KB shared memory, got %+v", el100)
		}
	})

	t.Run("TMEM target fencing sm_100a vs sm_121a and sm_90", func(t *testing.T) {
		prod := testProducerDesc("prod_tmem")
		prod.Resources = &cudakernel.ResourceRequirements{
			AllocatedRegisters: 32,
			TMEMBytes:          65536,
			ExecutionGroup:     "tcgen05",
		}
		cons := testConsumerDesc("cons")

		for _, unsupported := range []string{"sm_90", "sm_90a", "sm_100", "sm_121", "sm_121a"} {
			el := cudapdl.CheckEligibility(unsupported, prod, cons)
			if el.Eligible || !strings.Contains(el.FallbackReason, "does not support TMEM") {
				t.Fatalf("expected TMEM rejection on %s, got %+v", unsupported, el)
			}
		}

		el100a := cudapdl.CheckEligibility("sm_100a", prod, cons)
		if !el100a.Eligible || el100a.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("expected TMEM + tcgen05 admission on sm_100a, got %+v", el100a)
		}
	})

	t.Run("ExecutionGroup wgmma target fencing sm_90a vs sm_100a and sm_121a", func(t *testing.T) {
		prod := testProducerDesc("prod_wgmma")
		prod.Resources = &cudakernel.ResourceRequirements{
			AllocatedRegisters: 64,
			ExecutionGroup:     "wgmma",
		}
		cons := testConsumerDesc("cons")

		for _, unsupported := range []string{"sm_90", "sm_100a", "sm_121a"} {
			el := cudapdl.CheckEligibility(unsupported, prod, cons)
			if el.Eligible || !strings.Contains(el.FallbackReason, `ExecutionGroup "wgmma" is not supported on target`) {
				t.Fatalf("expected wgmma rejection on %s, got %+v", unsupported, el)
			}
		}

		el90a := cudapdl.CheckEligibility("sm_90a", prod, cons)
		if !el90a.Eligible || el90a.Policy != cudapdl.PolicyProgrammatic {
			t.Fatalf("expected wgmma admission on sm_90a, got %+v", el90a)
		}
	})

	t.Run("single-SM co-residency optional vs required", func(t *testing.T) {
		prod := testProducerDesc("prod_heavy_smem")
		prod.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 65536,
			AllocatedRegisters: 32,
		}
		cons := testConsumerDesc("cons_heavy_smem")
		cons.Resources = &cudakernel.ResourceRequirements{
			StaticSharedMemory: 65536, // combined 131,072 B > 102,400 B on sm_121a
			AllocatedRegisters: 32,
		}

		// Without RequireSingleSMCoResidency, individual CTAs fit on sm_121a (64 KiB <= 100 KiB),
		// SingleSMCoResidentPossible is reported false, and no-overlap correctness still admits the pair.
		elDefault := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if !elDefault.Eligible || elDefault.Resources.SingleSMCoResidentPossible {
			t.Fatalf("expected eligible with SingleSMCoResidentPossible=false on sm_121a, got %+v", elDefault)
		}

		// With RequireSingleSMCoResidency=true, exceeding single-SM combined shared memory falls back.
		cons.Effects.RequireSingleSMCoResidency = true
		elRequired := cudapdl.CheckEligibility("sm_121a", prod, cons)
		if elRequired.Eligible || !strings.Contains(elRequired.FallbackReason, "exceed single-SM co-residency limits on sm_121a") {
			t.Fatalf("expected single-SM co-residency fallback on sm_121a, got %+v", elRequired)
		}
	})

	for _, tc := range []struct {
		name       string
		target     string
		mutate     func(prod, cons *cudapdl.KernelDesc)
		wantReason string
	}{
		{
			name:   "missing producer resources",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources = nil
			},
			wantReason: "missing required resource requirements",
		},
		{
			name:   "missing consumer ThreadsPerBlock",
			target: "sm_90",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.ThreadsPerBlock = 0
			},
			wantReason: "missing required ThreadsPerBlock",
		},
		{
			name:   "ThreadsPerBlock exceeds 1024",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.ThreadsPerBlock = 1056
			},
			wantReason: "ThreadsPerBlock (1056) exceeds architectural limit 1024",
		},
		{
			name:   "zero AllocatedRegisters",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.AllocatedRegisters = 0
			},
			wantReason: "missing AllocatedRegisters",
		},
		{
			name:   "AllocatedRegisters exceeds 255",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.AllocatedRegisters = 256
			},
			wantReason: "AllocatedRegisters (256) exceeds per-thread limit 255",
		},
		{
			name:   "PeakLiveGPR exceeds AllocatedRegisters",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.AllocatedRegisters = 32
				prod.Resources.PeakLiveGPR = 48
			},
			wantReason: "PeakLiveGPR (48) exceeds AllocatedRegisters (32)",
		},
		{
			name:   "RegisterQuantum mismatch",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.RegisterQuantum = 512
			},
			wantReason: "RegisterQuantum (512) mismatches target sm_90 quantum (256)",
		},
		{
			name:   "CTA registers exceed 65536 per-SM limit",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.ThreadsPerBlock = 1024
				prod.Resources.AllocatedRegisters = 128 // 1024 * 128 = 131,072 > 65,536
			},
			wantReason: "CTA registers (131072) exceed per-SM register file limit 65536",
		},
		{
			name:   "static plus dynamic shared memory overflows uint32",
			target: "sm_90",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.StaticSharedMemory = math.MaxUint32 - 511
				prod.Resources.DynamicSharedMemory = 1024
			},
			wantReason: "shared memory overflows uint32",
		},
		{
			name:   "dynamic shared memory exceeds opt-in limit",
			target: "sm_121a",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.StaticSharedMemory = 0
				prod.Resources.DynamicSharedMemory = 102000 // > 101,376 opt-in on sm_121a
			},
			wantReason: "dynamic shared memory (102000 B) exceeds target sm_121a opt-in limit (101376 B)",
		},
		{
			name:   "TMEM per-kernel exceeds 256 KiB on sm_100a",
			target: "sm_100a",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.TMEMBytes = 300 * 1024
			},
			wantReason: "TMEMBytes (307200 B) exceeds 262144 B per-SM TMEM limit",
		},
		{
			name:   "combined producer and consumer TMEM exceeds 256 KiB on sm_100a",
			target: "sm_100a",
			mutate: func(prod, cons *cudapdl.KernelDesc) {
				prod.Resources.TMEMBytes = 192 * 1024
				cons.Resources.TMEMBytes = 128 * 1024
			},
			wantReason: "combined producer/consumer TMEM (327680 B) exceeds 262144 B per-SM TMEM limit",
		},
		{
			name:   "unrecognized ExecutionGroup",
			target: "sm_90",
			mutate: func(_, cons *cudapdl.KernelDesc) {
				cons.Resources.ExecutionGroup = "warp_specialized_magic"
			},
			wantReason: `unrecognized ExecutionGroup "warp_specialized_magic"`,
		},
		{
			name:   "ExecutionGroup tcgen05 on sm_90a refused",
			target: "sm_90a",
			mutate: func(prod, _ *cudapdl.KernelDesc) {
				prod.Resources.ExecutionGroup = "tcgen05"
			},
			wantReason: `ExecutionGroup "tcgen05" is not supported on target sm_90a`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prod := testProducerDesc("prod")
			cons := testConsumerDesc("cons")
			tc.mutate(&prod, &cons)

			el := cudapdl.CheckEligibility(tc.target, prod, cons)
			if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback {
				t.Fatalf("expected resource fallback, got %+v", el)
			}
			if !strings.Contains(el.FallbackReason, tc.wantReason) {
				t.Fatalf("FallbackReason = %q, want substring %q", el.FallbackReason, tc.wantReason)
			}
		})
	}
}
