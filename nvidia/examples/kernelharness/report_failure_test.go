package main

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

func TestClassifiedFailureEncodesForCLI(t *testing.T) {
	for _, value := range []float32{1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		c, err := cudahotswap.Compare([]float32{value}, []float32{0}, cudahotswap.Tolerance{})
		if err != nil {
			t.Fatal(err)
		}
		rep := cudahotswap.Report{Outcome: cudahotswap.OutcomeWrongAnswer, Correctness: &cudahotswap.CorrectnessReport{Trials: []cudahotswap.TrialReport{{Comparison: c}}}}
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(rep); err != nil {
			t.Fatalf("CLI would exit4 instead of classifiedfailure: %v", err)
		}
		if rep.Outcome.ExitCode() != 1 || c.OK() || !json.Valid(b.Bytes()) {
			t.Fatal("CLI failure became unclassified or success")
		}
	}
}
