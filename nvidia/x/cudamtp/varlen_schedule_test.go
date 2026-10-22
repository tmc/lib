// SPDX-License-Identifier: Apache-2.0

package cudamtp

import (
	"errors"
	"math"
	"sort"
	"testing"
)

func TestMedianTracker(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
	}{
		{
			name:   "single_element",
			values: []float64{42.5},
		},
		{
			name:   "odd_increasing",
			values: []float64{1, 2, 3, 4, 5, 6, 7},
		},
		{
			name:   "even_decreasing",
			values: []float64{80, 70, 60, 50, 40, 30, 20, 10},
		},
		{
			name:   "duplicates_and_negatives",
			values: []float64{5, -2, 5, 0, -2, 10, 5, 3, 3, 1},
		},
	}

	var totalChecked int
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mt := NewMedianTracker()
			if mt.Count() != 0 || mt.Median() != 0 {
				t.Fatalf("empty MedianTracker: got count=%d median=%v, want 0, 0", mt.Count(), mt.Median())
			}

			var prefix []float64
			for i, v := range tc.values {
				mt.Update(v)
				prefix = append(prefix, v)

				sorted := make([]float64, len(prefix))
				copy(sorted, prefix)
				sort.Float64s(sorted)
				var want float64
				mid := len(sorted) / 2
				if len(sorted)%2 == 1 {
					want = sorted[mid]
				} else {
					want = (sorted[mid-1] + sorted[mid]) * 0.5
				}

				got := mt.Median()
				if math.Abs(got-want) > 1e-12 {
					t.Fatalf("step %d: got median %v, want %v (prefix=%v)", i, got, want, prefix)
				}
				if mt.Count() != len(prefix) {
					t.Fatalf("step %d: got count %d, want %d", i, mt.Count(), len(prefix))
				}
				totalChecked++
			}

			mt.Clear()
			if mt.Count() != 0 || mt.Median() != 0 {
				t.Fatalf("after Clear: got count=%d median=%v, want 0, 0", mt.Count(), mt.Median())
			}
		})
	}
	if totalChecked < 20 {
		t.Fatalf("checked only %d median states; want >= 20", totalChecked)
	}
}

func TestForwardRuntimeEstimator(t *testing.T) {
	est := NewForwardRuntimeEstimator([]int{8, 16, 32})

	// Calibrate bucket 8 (base 2.0ms at K=1000, 3.0ms at K=2000 -> slope = 0.001 ms/tok)
	// and bucket 16 (base 4.0ms at K=1000, 5.0ms at K=2000 -> slope = 0.001 ms/tok).
	for i := 0; i < MinSamplesForReliableEstimate; i++ {
		k := 1000
		if i%2 == 1 {
			k = 2000
		}
		est.Record(PerfPoint{
			TotalSeqlensQ: 8,
			TotalSeqlensK: k,
			DurationMS:    1.0 + 0.001*float64(k),
		})
		est.Record(PerfPoint{
			TotalSeqlensQ: 16,
			TotalSeqlensK: k,
			DurationMS:    3.0 + 0.001*float64(k),
		})
	}

	if math.Abs(est.KVCacheSlope()-0.001) > 1e-6 {
		t.Fatalf("KVCacheSlope() = %v, want 0.001", est.KVCacheSlope())
	}

	// Linear interpolation at midpoint Q=12, K=1500:
	// Bucket 8 base at K=0 is 1.0ms; bucket 16 base at K=0 is 3.0ms -> Q=12 base is 2.0ms.
	// With K=1500 and slope=0.001, expected duration is 2.0 + 1.5 = 3.5ms.
	gotMid := est.Estimate(12, 1500)
	if math.Abs(gotMid-3.5) > 1e-4 {
		t.Fatalf("Estimate(12, 1500) = %v, want 3.5", gotMid)
	}

	// Verify monotonic increase with KV length.
	gotHighK := est.Estimate(12, 3500)
	if math.Abs(gotHighK-5.5) > 1e-4 {
		t.Fatalf("Estimate(12, 3500) = %v, want 5.5", gotHighK)
	}

	// Push across MaxWindowSize to verify two-tier (recent -> old) rollover.
	for i := 0; i < MaxWindowSize; i++ {
		est.Record(PerfPoint{
			TotalSeqlensQ: 8,
			TotalSeqlensK: 1000,
			DurationMS:    2.0,
		})
	}
	gotAfterRollover := est.Estimate(8, 1000)
	if gotAfterRollover <= 0 || math.IsNaN(gotAfterRollover) {
		t.Fatalf("invalid Estimate after window rollover: %v", gotAfterRollover)
	}
}

func TestSamplerRuntimeEstimator(t *testing.T) {
	s := NewSamplerRuntimeEstimator()
	// Feed 9 samples with outliers to verify 9-sample median filter and linear scaling.
	samples := []struct {
		tokens int
		ms     float64
	}{
		{8, 0.18},
		{8, 0.18},
		{8, 0.18},
		{8, 0.18},
		{16, 0.26},
		{16, 0.26},
		{16, 0.26},
		{16, 0.26},
		{16, 0.26},
	}
	for _, sm := range samples {
		s.Record(sm.tokens, sm.ms)
	}

	est8 := s.Estimate(8)
	est16 := s.Estimate(16)
	if math.Abs(est8-0.18) > 1e-5 {
		t.Fatalf("Estimate(8) = %v, want 0.18", est8)
	}
	if math.Abs(est16-0.26) > 1e-5 {
		t.Fatalf("Estimate(16) = %v, want 0.26", est16)
	}
	if !(est16 > est8) {
		t.Fatalf("Estimate(16)=%v must exceed Estimate(8)=%v", est16, est8)
	}
}

func TestVarlenScheduler(t *testing.T) {
	_, err := NewVarlenScheduler(VarlenScheduleConfig{NumSpecSteps: 0, Buckets: []int{4}})
	if !errors.Is(err, ErrInvalidVarlenConfig) {
		t.Fatalf("expected ErrInvalidVarlenConfig for NumSpecSteps=0, got %v", err)
	}

	sched, err := NewVarlenScheduler(VarlenScheduleConfig{
		NumSpecSteps:   3,
		Buckets:        []int{2, 4, 8},
		EWMADecay:      0.8,
		MinDynamicProb: 0.25,
		StepOverheadMS: 0.1,
	})
	if err != nil {
		t.Fatalf("NewVarlenScheduler: %v", err)
	}

	if _, err := sched.Schedule(nil); !errors.Is(err, ErrEmptySequenceBatch) {
		t.Fatalf("expected ErrEmptySequenceBatch, got %v", err)
	}

	// Calibrate bucket 2 at 1.0ms, bucket 4 at 1.15ms (nearly free extra tokens up to 4),
	// and bucket 8 at 10.0ms (steep latency wall so budget stops at 4).
	for i := 0; i < MinSamplesForReliableEstimate; i++ {
		sched.RecordStep(PerfPoint{TotalSeqlensQ: 2, TotalSeqlensK: 512, DurationMS: 1.0})
		sched.RecordStep(PerfPoint{TotalSeqlensQ: 4, TotalSeqlensK: 512, DurationMS: 1.15})
		sched.RecordStep(PerfPoint{TotalSeqlensQ: 8, TotalSeqlensK: 512, DurationMS: 10.0})
	}
	for i := 0; i < SamplerMedianWindowSize; i++ {
		sched.RecordSampler(4, 0.05)
	}

	seqs := []SequenceMTPState{
		{
			SeqID:       101,
			ContextLen:  256,
			AcceptProbs: []float64{0.95, 0.90, 0.20}, // cumProbs: 0.95, 0.855, 0.171
			EWMAProb:    0.85,
		},
		{
			SeqID:       102,
			ContextLen:  256,
			AcceptProbs: []float64{0.50, 0.40, 0.10}, // cumProbs: 0.50, 0.20, 0.02
			EWMAProb:    0.75,
		},
	}

	dec, err := sched.Schedule(seqs)
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}

	// Bucket 4 is optimal because 2 extra tokens fit inside the 1.15ms bucket,
	// whereas crossing >4 jumps to the 10.0ms bucket.
	if dec.TargetTotalQ != 4 || dec.BucketBsz != 4 {
		t.Fatalf("got TargetTotalQ=%d BucketBsz=%d, want 4, 4", dec.TargetTotalQ, dec.BucketBsz)
	}
	// Seq 0 has the top two marginal cumulative probabilities (0.95 and 0.855 > 0.50),
	// so Seq 0 gets 1+2=3 tokens and Seq 1 gets 1 token.
	if len(dec.SeqlensQ) != 2 || dec.SeqlensQ[0] != 3 || dec.SeqlensQ[1] != 1 {
		t.Fatalf("got SeqlensQ=%v, want [3 1]", dec.SeqlensQ)
	}
	wantExpected := 2.0 + 0.95 + 0.855
	if math.Abs(dec.EstimatedExpectedAccept-wantExpected) > 1e-6 {
		t.Fatalf("got EstimatedExpectedAccept=%v, want %v", dec.EstimatedExpectedAccept, wantExpected)
	}

	// Verify UpdateAcceptance EWMA rule: 0.8*0.85 + 0.2*(3/4) = 0.68 + 0.15 = 0.83.
	sched.UpdateAcceptance(101, 4, 3)
	gotEWMA, ok := sched.EWMAProb(101)
	if !ok || math.Abs(gotEWMA-0.83) > 1e-9 {
		t.Fatalf("after UpdateAcceptance: got (%v, %v), want (0.83, true)", gotEWMA, ok)
	}
}
