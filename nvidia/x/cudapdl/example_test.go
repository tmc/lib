package cudapdl_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func ExampleCheckEligibility() {
	producer := cudapdl.KernelDesc{
		Name:                "rms_norm",
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
				{Kind: cudapdl.EffectMemory, Resource: "x", Access: cudapdl.AccessRead, Bytes: 4096},
				{Kind: cudapdl.EffectMemory, Resource: "normed", Access: cudapdl.AccessWrite, Bytes: 4096},
			},
			AfterLaunchDependents: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectPureCompute},
			},
		},
	}
	consumer := cudapdl.KernelDesc{
		Name:               "gemv_projection",
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
				{Kind: cudapdl.EffectMemory, Resource: "w_proj", Access: cudapdl.AccessRead, Bytes: 16384},
			},
			AfterWait: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "normed", Access: cudapdl.AccessRead, Bytes: 4096},
				{Kind: cudapdl.EffectMemory, Resource: "out", Access: cudapdl.AccessWrite, Bytes: 4096},
			},
		},
	}

	el := cudapdl.CheckEligibility("sm_90", producer, consumer)
	fmt.Printf("sm_90 eligible: %v (policy: %s)\n", el.Eligible, el.Policy)

	// Architecture below sm_90
	elLegacy := cudapdl.CheckEligibility("sm_80", producer, consumer)
	fmt.Printf("sm_80 eligible: %v (policy: %s)\n", elLegacy.Eligible, elLegacy.Policy)

	// Output:
	// sm_90 eligible: true (policy: PolicyProgrammatic)
	// sm_80 eligible: false (policy: PolicyStandardFallback)
}

func ExamplePlanAdapter() {
	adapter := cudapdl.NewPlanAdapter("sm_100a")

	prod := cudapdl.KernelDesc{
		Name:                "gemv_stage1",
		HasLaunchDependents: true,
		ThreadsPerBlock:     256,
		Resources: &cudakernel.ResourceRequirements{
			StaticSharedMemory: 8192,
			AllocatedRegisters: 40,
			PeakLiveGPR:        32,
		},
		Effects: cudapdl.KernelEffects{
			TriggerPlacement: cudapdl.TriggerAfterSharedWrites,
			BeforeLaunchDependents: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "x", Access: cudapdl.AccessRead, Bytes: 8192},
				{Kind: cudapdl.EffectMemory, Resource: "stage1_out", Access: cudapdl.AccessWrite, Bytes: 8192},
			},
		},
	}
	cons := cudapdl.KernelDesc{
		Name:               "gemv_stage2",
		HasWait:            true,
		HasIndependentWork: true,
		ThreadsPerBlock:    256,
		Resources: &cudakernel.ResourceRequirements{
			StaticSharedMemory: 8192,
			AllocatedRegisters: 40,
			PeakLiveGPR:        32,
		},
		Effects: cudapdl.KernelEffects{
			WaitPlacement: cudapdl.WaitAfterIndependentPreamble,
			BeforeWait: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "w2", Access: cudapdl.AccessRead, Bytes: 32768},
			},
			AfterWait: []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "stage1_out", Access: cudapdl.AccessRead, Bytes: 8192},
				{Kind: cudapdl.EffectMemory, Resource: "y", Access: cudapdl.AccessWrite, Bytes: 8192},
			},
		},
	}

	el := adapter.Evaluate(prod, cons)
	fmt.Printf("policy: %s\n", el.Policy)

	// Output:
	// policy: PolicyProgrammatic
}
