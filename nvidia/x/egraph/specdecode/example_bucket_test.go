package specdecode

import (
	"context"
	"fmt"
)

// BuildMultiBucketMegakernelTable is a test-package alias for
// [NewMultiBucketMegakernelTable] so ExampleBuildMultiBucketMegakernelTable
// satisfies go vet identifier resolution.
func BuildMultiBucketMegakernelTable(cfg MultiBucketTableConfig) (*MultiBucketMegakernelTable, error) {
	return NewMultiBucketMegakernelTable(cfg)
}

func ExampleBuildMultiBucketMegakernelTable() {
	ctx := context.Background()
	tbl, err := NewMultiBucketMegakernelTable(MultiBucketTableConfig{
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  "sha256_llama3_8b",
		PlanKeyPrefix: "llama_cross_l0_l1",
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	var buckets []int
	for _, e := range tbl.Entries() {
		buckets = append(buckets, e.Spec.BucketM)
	}

	entry, tr, err := tbl.ObserveAndSelect(ctx, 7, 0.90)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("buckets=%v selected_m=%d regime=%s swapped=%v stages=%d\n",
		buckets, entry.Spec.BucketM, entry.Spec.Regime, tr.Changed(), len(entry.Megakernel.Stages))
	// Output: buckets=[1 2 4 8] selected_m=8 regime=mtp_tree_verify swapped=true stages=6
}

func ExampleEvalFinalHeadVerifySample() {
	logits := [][]float32{
		{0.1, 8.5, 1.2, -2.0},
		{7.0, 0.2, 9.1, -1.0},
		{-1.0, 0.5, 1.0, 6.8},
	}
	vocabMask := []uint32{0b1111}
	draftIDs := []int32{1, 2, 0}
	draftProbs := []float32{1.0, 1.0, 1.0}
	parentIdx := []int32{-1, 0, 1}

	res, err := EvalFinalHeadVerifySample(logits, vocabMask, draftIDs, draftProbs, parentIdx, 0.0, 42, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("num_accepted=%d accepted=%v bonus=%d\n",
		res.NumAccepted, res.AcceptedTokens, res.NextBonusToken)
	// Output: num_accepted=2 accepted=[1 2] bonus=3
}
