// SPDX-License-Identifier: Apache-2.0

package cudamtp_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamtp"
)

func ExampleMedianTracker() {
	mt := cudamtp.NewMedianTracker()
	for _, v := range []float64{1.8, 1.2, 2.4, 1.5, 1.9} {
		mt.Update(v)
	}
	fmt.Printf("count=%d median=%.1f\n", mt.Count(), mt.Median())
	// Output:
	// count=5 median=1.8
}

func ExampleVarlenScheduler() {
	sched, err := cudamtp.NewVarlenScheduler(cudamtp.VarlenScheduleConfig{
		NumSpecSteps:   2,
		Buckets:        []int{2, 4, 8},
		StepOverheadMS: 0.05,
	})
	if err != nil {
		panic(err)
	}
	for i := 0; i < cudamtp.MinSamplesForReliableEstimate; i++ {
		sched.RecordStep(cudamtp.PerfPoint{TotalSeqlensQ: 2, TotalSeqlensK: 256, DurationMS: 1.0})
		sched.RecordStep(cudamtp.PerfPoint{TotalSeqlensQ: 4, TotalSeqlensK: 256, DurationMS: 1.1})
		sched.RecordStep(cudamtp.PerfPoint{TotalSeqlensQ: 8, TotalSeqlensK: 256, DurationMS: 5.0})
	}
	decision, err := sched.Schedule([]cudamtp.SequenceMTPState{
		{SeqID: 1, ContextLen: 128, AcceptProbs: []float64{0.9, 0.8}, EWMAProb: 0.85},
		{SeqID: 2, ContextLen: 128, AcceptProbs: []float64{0.4, 0.2}, EWMAProb: 0.60},
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("totalQ=%d bucket=%d seqlensQ=%v\n", decision.TargetTotalQ, decision.BucketBsz, decision.SeqlensQ)
	// Output:
	// totalQ=4 bucket=4 seqlensQ=[3 1]
}
