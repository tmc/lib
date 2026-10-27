package autotune_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

func ExampleOnlineCostModel() {
	twin, err := autotune.AutoDiscoverTarget(10, 7, true, true)
	if err != nil {
		panic(err)
	}

	candidates := []autotune.ScheduleCandidate{
		{
			Name:             "tcgen05_2stage_dense",
			Target:           twin.Target,
			TileM:            128,
			TileN:            256,
			TileK:            64,
			Stages:           2,
			CTAGroup:         1,
			AnalyticalCostUS: 12.0,
		},
		{
			Name:             "tcgen05_4stage_lut_sync_restrict",
			Target:           twin.Target,
			TileM:            128,
			TileN:            256,
			TileK:            128,
			Stages:           4,
			CTAGroup:         1,
			UseLUTB:          true,
			UseEarlyRelease:  true,
			AnalyticalCostUS: 13.5,
		},
	}

	model := autotune.NewOnlineCostModel()
	const shape = "M=128,N=4096,K=14336"

	model.RecordObservation(
		shape,
		candidates[0],
		autotune.StallProfile{
			TotalSamples:        1000,
			StallLongScoreboard: 650,
			StallNotSelected:    350,
		},
		autotune.RangeMetrics{
			DurationNS:    24000,
			RegisterCount: 168,
		},
	)

	best, rec, err := model.SelectBest(shape, twin, candidates)
	if err != nil {
		panic(err)
	}
	fmt.Printf("target=%s tmem=%d best=%s bottleneck=%s stages=%d early_release=%v\n",
		twin.Target, twin.TMEMColumns, best.Name, rec.Bottleneck, rec.SuggestedStages, rec.EnableEarlyReleaseSyncRestrict)
	// Output:
	// target=sm_107f tmem=576 best=tcgen05_4stage_lut_sync_restrict bottleneck=memory_latency stages=4 early_release=true
}
