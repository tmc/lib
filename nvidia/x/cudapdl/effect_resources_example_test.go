package cudapdl_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func ExampleCheckEligibility_phaseEffectsAndResources() {
	producer := cudapdl.KernelDesc{
		Name:                "rmsnorm_producer",
		HasLaunchDependents: true,
		ThreadsPerBlock:     128,
		Resources: &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 32,
			PeakLiveGPR:        24,
		},
		Effects: cudapdl.KernelEffects{
			TriggerPlacement: cudapdl.TriggerAfterSharedWrites,
			BeforeLaunchDependents: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "activations", Access: cudapdl.AccessRead, Bytes: 4096},
				{Kind: cudapdl.EffectMemory, Resource: "normed_out", Access: cudapdl.AccessWrite, Bytes: 4096},
			},
			AfterLaunchDependents: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectPureCompute},
			},
		},
	}

	consumer := cudapdl.KernelDesc{
		Name:               "gemv_consumer",
		HasWait:            true,
		HasIndependentWork: true,
		ThreadsPerBlock:    128,
		Resources: &cudakernel.ResourceRequirements{
			StaticSharedMemory: 4096,
			AllocatedRegisters: 32,
			PeakLiveGPR:        24,
		},
		Effects: cudapdl.KernelEffects{
			WaitPlacement: cudapdl.WaitAfterIndependentPreamble,
			BeforeWait: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "proj_weights", Access: cudapdl.AccessRead, Bytes: 16384},
			},
			AfterWait: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "normed_out", Access: cudapdl.AccessRead, Bytes: 4096},
				{Kind: cudapdl.EffectMemory, Resource: "proj_out", Access: cudapdl.AccessWrite, Bytes: 4096},
			},
		},
	}

	okEl := cudapdl.CheckEligibility("sm_121a", producer, consumer)
	fmt.Printf("safe pair: eligible=%v policy=%s provenance=%s preamble=%v overlapGuaranteed=%v\n",
		okEl.Eligible, okEl.Policy, okEl.Provenance, okEl.ConsumerHasIndependentPreamble, okEl.OverlapGuaranteed)

	// Prematurely reading the producer handoff buffer in BeforeWait is refused.
	badConsumer := consumer
	badConsumer.Effects.BeforeWait = []cudapdl.PhaseEffect{
		{Kind: cudapdl.EffectMemory, Resource: "normed_out", Access: cudapdl.AccessRead, Bytes: 4096},
	}
	hazardEl := cudapdl.CheckEligibility("sm_121a", producer, badConsumer)
	fmt.Printf("premature read: eligible=%v policy=%s\n", hazardEl.Eligible, hazardEl.Policy)

	// Output:
	// safe pair: eligible=true policy=PolicyProgrammatic provenance=ProvenanceOwnerDeclared preamble=true overlapGuaranteed=false
	// premature read: eligible=false policy=PolicyStandardFallback
}

func ExamplePlanAdapter_Evaluate_resourceAndEffectFallback() {
	adapter := cudapdl.NewPlanAdapter("sm_121a")

	producer := cudapdl.KernelDesc{
		Name:                "tmem_producer",
		HasLaunchDependents: true,
		ThreadsPerBlock:     128,
		Resources: &cudakernel.ResourceRequirements{
			AllocatedRegisters: 32,
			TMEMBytes:          16384,
		},
		Effects: cudapdl.KernelEffects{
			TriggerPlacement: cudapdl.TriggerAfterSharedWrites,
			BeforeLaunchDependents: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "handoff", Access: cudapdl.AccessWrite, Bytes: 4096},
			},
		},
	}
	consumer := cudapdl.KernelDesc{
		Name:            "entry_sync_consumer",
		HasWait:         true,
		ThreadsPerBlock: 128,
		Resources: &cudakernel.ResourceRequirements{
			AllocatedRegisters: 32,
		},
		Effects: cudapdl.KernelEffects{
			WaitPlacement: cudapdl.WaitAtEntryBeforeDependentBody,
			AfterWait: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "handoff", Access: cudapdl.AccessRead, Bytes: 4096},
				{Kind: cudapdl.EffectMemory, Resource: "out", Access: cudapdl.AccessWrite, Bytes: 4096},
			},
		},
	}

	// sm_121a (GB10) does not support TMEM; adapter falls back to ordinary dependency.
	el121 := adapter.Evaluate(producer, consumer)
	fmt.Printf("sm_121a policy: %s (eligible=%v)\n", el121.Policy, el121.Eligible)

	// sm_100a supports TMEM; entry-sync consumer is admitted without independent preamble.
	el100 := cudapdl.NewPlanAdapter("sm_100a").Evaluate(producer, consumer)
	fmt.Printf("sm_100a policy: %s (eligible=%v preamble=%v)\n",
		el100.Policy, el100.Eligible, el100.ConsumerHasIndependentPreamble)

	// Output:
	// sm_121a policy: PolicyStandardFallback (eligible=false)
	// sm_100a policy: PolicyProgrammatic (eligible=true preamble=false)
}
