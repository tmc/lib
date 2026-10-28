package cudaspec_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

func ExampleBatchDraftProducer() {
	a := cudakv.NewBlockAllocator(2, 8)
	if err := a.AllocSequence(10, nil, 2); err != nil {
		fmt.Println(err)
		return
	}
	req := cudaspec.BatchCycleRequest{
		Prefix:         []int32{0, 1},
		PrefixSequence: 10,
		Sequence:       11,
		Hidden:         []float32{1},
		DraftWeights:   []float32{1, 1},
		HeadWeights:    []float32{0, 0, 2, 0, 0, 0, 0, 3},
		MTP:            cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 1, VocabSize: 4},
	}
	var model [32]byte
	model[0] = 1
	req.Key = cudaspec.ComputeSpeculationKey(model[:], req.Prefix, 1, 2, 2, "fp32", 1)
	req.ExpectedKey = req.Key
	req.DraftDigest, _ = cudaspec.MTPDraftDigest(req.MTP, req.DraftWeights, req.HeadWeights)
	req.DraftProducer = &cudaspec.BatchDraftProducer{
		Engine: cudaspec.DraftLogitsFunc(func(_ context.Context, step *cudaspec.BatchDraftStep) ([]float32, error) {
			return cudamtp.MTPReference(step.Hidden, step.DraftWeights, step.HeadWeights, step.MTP)
		}),
		Wait: func(*cudaspec.BatchDraftStep) error { return nil },
	}
	v := cudaspec.BatchVerifier{
		ReadOnlyPrefix: true,
		Engine:         cudabatch.NewHostDeterministicEngine(0, []int32{2, 3, 1}),
		Wait:           func(*cudabatch.ExecutionStep) error { return nil },
	}
	result, err := cudaspec.ExecuteBatchCycle(context.Background(), a, req, v)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Outcome.DraftTokens, result.Outcome.AcceptedTokens, result.Outcome.BonusToken, result.Outcome.Committed)
	_ = result.KV.Close()
	// Output:
	// [2 3] [2 3] 1 true
}
