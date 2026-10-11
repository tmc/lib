package cudaspec_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

func ExampleExecuteBatchCycle() {
	// Host-only fixture. An actual model engine needs separate native admission.
	a := cudakv.NewBlockAllocator(2, 8)
	if err := a.AllocSequence(10, nil, 2); err != nil {
		fmt.Println(err)
		return
	}
	req := cudaspec.BatchCycleRequest{Prefix: []int32{0, 1}, PrefixSequence: 10, Sequence: 11, Hidden: []float32{1}, DraftWeights: []float32{1, 1}, HeadWeights: []float32{0, 0, 2, 0, 0, 0, 0, 3}, MTP: cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 1, VocabSize: 4}}
	var model [32]byte
	model[0] = 1
	req.Key = cudaspec.ComputeSpeculationKey(model[:], req.Prefix, 1, 2, 2, "fp32", 1)
	req.ExpectedKey = req.Key
	req.DraftDigest, _ = cudaspec.MTPDraftDigest(req.MTP, req.DraftWeights, req.HeadWeights)
	v := cudaspec.BatchVerifier{ReadOnlyPrefix: true, Engine: cudabatch.NewHostDeterministicEngine(0, []int32{2, 3, 1}), Wait: func(*cudabatch.ExecutionStep) error { return nil }}
	result, err := cudaspec.ExecuteBatchCycle(context.Background(), a, req, v)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Outcome.AcceptedTokens, result.Outcome.BonusToken, result.Outcome.Committed)
	if err := result.KV.Close(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.NumFreeBlocks())
	// Output:
	// [2 3] 1 true
	// 7
}
func ExampleNewBranchKV() {
	a := cudakv.NewBlockAllocator(2, 4)
	_ = a.AllocSequence(10, nil, 2)
	branch, err := cudaspec.NewBranchKV(a, 10, 11, 2, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	table, err := branch.Table()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(table))
	_ = branch.Close()
	fmt.Println(a.NumFreeBlocks())
	// Output:
	// 2
	// 3
}
