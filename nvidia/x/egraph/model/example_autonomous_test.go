package model_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

func ExampleContinuousBatchScheduler() {
	ctx := context.Background()
	tree, err := model.NewRadixKVTree(32, 4)
	if err != nil {
		fmt.Println(err)
		return
	}
	table, err := specdecode.NewMultiBucketMegakernelTable(specdecode.MultiBucketTableConfig{
		Arch: "sm_121a",
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	sched, err := model.NewContinuousBatchScheduler(tree, table, 8)
	if err != nil {
		fmt.Println(err)
		return
	}

	if err := sched.Admit(&model.InFlightSequence{
		ID:           "req_decode",
		Phase:        model.PhaseDecode,
		PromptTokens: []int{10, 11, 12, 13},
		DecodeToken:  101,
	}); err != nil {
		fmt.Println(err)
		return
	}
	if err := sched.Admit(&model.InFlightSequence{
		ID:           "req_prefill",
		Phase:        model.PhaseChunkedPrefill,
		PromptTokens: []int{10, 11, 12, 13, 20, 21},
		ChunkBudget:  2,
		DecodeToken:  202,
	}); err != nil {
		fmt.Println(err)
		return
	}

	step, err := sched.ScheduleStep(ctx)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("active=%d bucket=%d padded=%d mask=%04b prefix_hits=%d row3_write_kv=%v\n",
		step.ActiveTokens,
		step.BucketM,
		step.PaddedRows,
		step.ActiveRowMask,
		step.PrefixCacheHits,
		step.Rows[3].WriteKV)
	// Output: active=3 bucket=4 padded=1 mask=0111 prefix_hits=1 row3_write_kv=false
}

func ExampleNewDisaggregatedPDEngine() {
	eng, err := model.NewDisaggregatedPDEngine(model.PDEngineConfig{
		Arch: model.ArchConfig{
			HiddenSize:        512,
			IntermediateSize:  1024,
			NumHiddenLayers:   4,
			NumAttentionHeads: 8,
			NumKeyValueHeads:  2,
			HeadDim:           64,
		},
		Generation:      nvl72.GenGB300,
		TotalRacks:      2,
		CacheKind:       nvl72.CacheMLA,
		KVLoRARank:      512,
		RoPEDim:         64,
		PageSize:        16,
		HBMPageCapacity: 8,
		C2CPageCapacity: 8,
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	page0 := make([]int32, 16)
	page1 := make([]int32, 16)
	for i := range 16 {
		page0[i] = int32(100 + i)
		page1[i] = int32(200 + i)
	}

	if _, err := eng.AdmitAndPrefill(model.PDRequest{
		ReqID:               "turn-1",
		TokenPages:          [][]int32{page0},
		ReleaseAfterPrefill: true,
	}); err != nil {
		fmt.Println(err)
		return
	}
	res2, err := eng.AdmitAndPrefill(model.PDRequest{
		ReqID:               "turn-2",
		TokenPages:          [][]int32{page0, page1},
		ReleaseAfterPrefill: true,
		ActiveDecodes:       []string{"dec-0"},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("hbm_hits=%d new_pages=%d saved_tokens=%d buckets=%v\n",
		res2.Receipt.HBMHitPages,
		res2.Receipt.NewlyAllocatedPages,
		res2.Receipt.PrefillTokensSaved,
		eng.DecodeBuckets())
	// Output: hbm_hits=1 new_pages=1 saved_tokens=16 buckets=[1 2 4 8]
}
