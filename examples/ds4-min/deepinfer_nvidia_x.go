package main

import (
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudaquant"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

// ds4GraphBatchBuckets builds piecewise CUDA Graph batch-size buckets up to
// maxBsz using github.com/tmc/lib/nvidia/x/cudagraph.
func ds4GraphBatchBuckets(maxBsz int) ([]int, error) {
	return cudagraph.BuildBatchBuckets(cudagraph.BucketPolicy{
		MaxBsz:    maxBsz,
		SmallStep: 4,
		MedStep:   8,
		LargeStep: 16,
	})
}

// ds4ResolveGraphBucket maps an active batch size to its padded CUDA Graph bucket.
func ds4ResolveGraphBucket(buckets []int, actualBsz int) (cudagraph.BucketPadding, error) {
	return cudagraph.ResolveBucket(buckets, actualBsz)
}

// ds4ConfidenceVarlenSchedule computes the confidence-ranked varlen seqlens_q
// and cu_seqlens_q verification schedule across speculative MTP draft steps.
func ds4ConfidenceVarlenSchedule(prob [][]float32, remap []int32, targetTotalBsz, maxStep int) (seqlensQ, cuSeqlensQ []int32, err error) {
	return cudaspec.ComputeSeqlensQByConfidenceCPU(prob, remap, targetTotalBsz, maxStep)
}

// ds4ZOSRemapPlan computes the Zero-Overhead Speculation (ZOS) device remap
// and position schedule between adjacent speculative steps without host sync.
func ds4ZOSRemapPlan(prevCuSeqlensQ, acceptedCounts, nextSeqlensQ, basePositions []int32) (cudabatch.ZOSRemapPlan, error) {
	return cudabatch.BuildZOSRemapPlan(prevCuSeqlensQ, acceptedCounts, nextSeqlensQ, basePositions)
}

// ds4PackUE8M0Scales quantizes and packs 4 UE8M0 power-of-two block exponents
// per uint32 word via github.com/tmc/lib/nvidia/x/cudaquant.
func ds4PackUE8M0Scales(amax []float32, fp8Max float32) ([]uint32, error) {
	exps := make([]uint8, len(amax))
	for i, v := range amax {
		exps[i] = cudaquant.QuantizeScaleUE8M0(v, fp8Max)
	}
	return cudaquant.PackUE8M0(exps)
}

// ds4VerifyBatchInvariance verifies bitwise float32 batch invariance across solo
// and batched executions via github.com/tmc/lib/nvidia/x/cudaagreement.
func ds4VerifyBatchInvariance(inputs [][]float32, batchSizes []int, fn func(batch [][]float32) ([][]float32, error)) (cudaagreement.BatchInvarianceReport, error) {
	return cudaagreement.VerifyBatchInvariance(inputs, batchSizes, fn)
}
