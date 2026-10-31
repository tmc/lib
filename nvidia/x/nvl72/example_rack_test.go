package nvl72_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvl72"
)

func ExampleCompileRackEPMegakernel() {
	plan, err := nvl72.CompileRackEPMegakernel(nvl72.RackEPMegakernelConfig{
		Name:            "gb300_rack_ep",
		Generation:      nvl72.GenGB300,
		NumRacks:        2,
		TokensPerRank:   64,
		HiddenSize:      4096,
		TopK:            8,
		TotalExperts:    144,
		TotalCTAs:       148,
		ThreadsPerCTA:   256,
		Microbatches:    2,
		EnableC2CHybrid: true,
		ColdTokenCount:  1,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("arch=%s gpus=%d ctas=%d+%d+%d c2c_ctas=%d cold_arm=%v speedup>1=%v\n",
		plan.TargetArch,
		plan.TotalGPUs,
		plan.ComputeCTAs,
		plan.DispatchCTAs,
		plan.CombineCTAs,
		plan.C2CDoorbellCTAs,
		plan.ColdPlacement.RunOnArmInPlace,
		plan.TBOSpeedup > 1.0)
	// Output: arch=sm_103a gpus=144 ctas=58+46+44 c2c_ctas=1 cold_arm=true speedup>1=true
}

func ExamplePlanMultiRackHierarchicalCollective() {
	spec := nvl72.MultiRackClusterSpec{
		NumRacks:             8,
		GPUsPerRack:          72,
		IntraRackBWGBps:      900.0,
		InterRackRDMA_BWGBps: 100.0,
		EnableIBGDA:          true,
		EnableSHARPv4:        true,
		StragglerDeadlineNs:  50_000,
	}
	plan, err := nvl72.PlanMultiRackHierarchicalCollective(spec, 64<<20)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("gpus=%d depth=%d inter=%s dispatch=%s speedup>1=%v\n",
		plan.TotalGPUs,
		plan.SHARPTreeDepth,
		plan.InterRackAlgorithm,
		plan.DispatchMechanism,
		plan.SpeedupFactor > 1.0)
	// Output: gpus=576 depth=3 inter=quantum3_spectrumx_sharp_v4_tree dispatch=ibgda_gpu_mmio_wqe_doorbell speedup>1=true
}
