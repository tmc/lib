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

func ExampleKnobPreamble() {
	twin, err := autotune.TwinForTarget("sm_100a")
	if err != nil {
		panic(err)
	}
	src, err := autotune.KnobPreamble(twin, autotune.ScheduleCandidate{
		Name:            "tma4_cta1",
		Stages:          4,
		CTAGroup:        1,
		ProducerMaxRegs: 40,
		ConsumerMaxRegs: 232,
	})
	if err != nil {
		panic(err)
	}
	fmt.Print(src)
	// Output:
	// // autotune schedule tma4_cta1 for sm_100a
	// #define AUTOTUNE_TMA_STAGES 4
	// #define AUTOTUNE_CTA_GROUP 1
	// #define AUTOTUNE_SWIZZLE_128B 0
	// #define AUTOTUNE_DOUBLE_BUFFER_TMEM 0
	// #define AUTOTUNE_SPARSE_2TO4 0
	// #define AUTOTUNE_PRODUCER_MAXNREG 40
	// __device__ __forceinline__ void autotune_producer_regs() { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory"); }
	// #define AUTOTUNE_CONSUMER_MAXNREG 232
	// __device__ __forceinline__ void autotune_consumer_regs() { asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory"); }
}
