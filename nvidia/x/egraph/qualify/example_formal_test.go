package qualify_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
)

func ExampleVerifyMegakernelBarrierSafety() {
	_, mega, err := hopper.BuildFullLayerMegakernel("formal_1l_mega", []hopper.FullDecoderLayerSpec{
		{
			LayerIndex: 0,
			BatchM:     1,
			HiddenDim:  256,
			IntermDim:  512,
			NumQHeads:  4,
			NumKVHeads: 2,
			HeadDim:    64,
			PageSize:   16,
			MaxPages:   4,
			Arch:       hopper.TargetSM121A,
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	rep, err := qualify.VerifyMegakernelBarrierSafety(mega)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("passed=%v stages=%d raw_ok=%v arena_ok=%v bar_ok=%v epoch_ok=%v\n",
		rep.Passed(),
		rep.NumStages,
		rep.RAWBarrierDominanceOK,
		rep.ArenaAliasingSafetyOK,
		rep.NamedBarrierUniformityOK,
		rep.EpochSlotNonOverlapOK)
	// Output: passed=true stages=9 raw_ok=true arena_ok=true bar_ok=true epoch_ok=true
}

func ExampleNewCalibratedCostModel() {
	cal := qualify.NewCalibratedCostModel(nil)
	cal.Observe(qualify.CostObservation{
		Op:                   hopper.OpGEMM,
		ScheduleKey:          "wgmma=false",
		AnalyticalCost:       2000,
		MeasuredHoldoutRatio: 1.50,
		RegistersPerThread:   96,
		TotalSpillBytes:      0,
	})
	cal.Observe(qualify.CostObservation{
		Op:                   hopper.OpGEMM,
		ScheduleKey:          "wgmma=true",
		AnalyticalCost:       2000,
		MeasuredHoldoutRatio: 0.80,
		RegistersPerThread:   224,
		TotalSpillBytes:      64,
	})

	fastMult := cal.Multiplier(hopper.OpGEMM, "wgmma=false")
	spilledMult := cal.Multiplier(hopper.OpGEMM, "wgmma=true")
	unseenMult := cal.Multiplier("unseen_op", "unseen_sched")
	fmt.Printf("fast<1=%v spilled>1=%v unseen=%.1f\n",
		fastMult < 1.0, spilledMult > 1.0, unseenMult)
	// Output: fast<1=true spilled>1=true unseen=1.0
}
