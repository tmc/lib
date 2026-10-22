package main

import "testing"

func TestDS4DeepInferNvidiaXIntegration(t *testing.T) {
	buckets, err := ds4GraphBatchBuckets(32)
	if err != nil {
		t.Fatalf("ds4GraphBatchBuckets failed: %v", err)
	}
	pad, err := ds4ResolveGraphBucket(buckets, 5)
	if err != nil {
		t.Fatalf("ds4ResolveGraphBucket failed: %v", err)
	}
	if pad.BucketBsz != 8 || pad.PadCount != 3 {
		t.Fatalf("unexpected bucket padding: %+v", pad)
	}

	probs := [][]float32{
		{0.95, 0.85, 0.40},
		{0.90, 0.30, 0.10},
	}
	seqlensQ, cuSeqlensQ, err := ds4ConfidenceVarlenSchedule(probs, []int32{0, 1}, 5, 3)
	if err != nil {
		t.Fatalf("ds4ConfidenceVarlenSchedule failed: %v", err)
	}
	if len(seqlensQ) != 2 || len(cuSeqlensQ) != 3 || cuSeqlensQ[2] != 5 {
		t.Fatalf("unexpected varlen schedule: seqlensQ=%v cuSeqlensQ=%v", seqlensQ, cuSeqlensQ)
	}

	zos, err := ds4ZOSRemapPlan([]int32{0, 3, 5}, []int32{2, 1}, seqlensQ, []int32{10, 20})
	if err != nil {
		t.Fatalf("ds4ZOSRemapPlan failed: %v", err)
	}
	if len(zos.Remap) != 2 || zos.Remap[0] != 1 || zos.Remap[1] != 3 {
		t.Fatalf("unexpected ZOS remap: %+v", zos)
	}

	packed, err := ds4PackUE8M0Scales([]float32{448.0, 224.0, 112.0, 56.0}, 448.0)
	if err != nil {
		t.Fatalf("ds4PackUE8M0Scales failed: %v", err)
	}
	if len(packed) != 1 || packed[0] == 0 {
		t.Fatalf("unexpected packed UE8M0 scales: %v", packed)
	}

	inputs := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{-1.5, 0.5, 2.5, -0.5},
	}
	report, err := ds4VerifyBatchInvariance(inputs, []int{2, 4}, func(batch [][]float32) ([][]float32, error) {
		out := make([][]float32, len(batch))
		for i, row := range batch {
			out[i] = make([]float32, len(row))
			copy(out[i], row)
		}
		return out, nil
	})
	if err != nil {
		t.Fatalf("ds4VerifyBatchInvariance failed: %v", err)
	}
	if !report.Passed() {
		t.Fatalf("expected batch invariance report to pass: %+v", report)
	}
}
