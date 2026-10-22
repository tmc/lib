package dsv4bench

import (
	"math"
	"strings"
	"testing"
)

func TestCompareAlignsRowsAndReportsPairedDelta(t *testing.T) {
	base := []Row{
		{CtxTokens: 2048, PrefillTokens: 2048, GenTokens: 128, GenSteadyTokens: 127, PrefillTPS: 10, GenTPS: 2, GenFirstMS: 5, GenSteadyTPS: 3},
		{CtxTokens: 4096, PrefillTokens: 2048, GenTokens: 128, GenSteadyTokens: 127, PrefillTPS: 20, GenTPS: 4, GenFirstMS: 4, GenSteadyTPS: 5},
	}
	target := []Row{
		{CtxTokens: 2048, PrefillTokens: 2048, GenTokens: 128, GenSteadyTokens: 127, PrefillTPS: 12, GenTPS: 1, GenFirstMS: 7, GenSteadyTPS: 4},
		{CtxTokens: 4096, PrefillTokens: 2048, GenTokens: 128, GenSteadyTokens: 127, PrefillTPS: 18, GenTPS: 5, GenFirstMS: 8, GenSteadyTPS: 6},
	}
	got, err := Compare(base, target)
	if err != nil {
		t.Fatal(err)
	}
	if got.Samples != 2 || len(got.CtxTokens) != 2 || got.CtxTokens[1] != 4096 {
		t.Fatalf("shape = %+v", got)
	}
	if got.PrefillTPS.Base != (Range{Min: 10, Median: 15, Max: 20}) ||
		got.PrefillTPS.Target != (Range{Min: 12, Median: 15, Max: 18}) ||
		got.PrefillTPS.Delta != (Range{Min: -2, Median: 0, Max: 2}) {
		t.Fatalf("prefill comparison = %+v", got.PrefillTPS)
	}
	if got.GenFirstMS.Delta != (Range{Min: 2, Median: 3, Max: 4}) {
		t.Fatalf("first-token delta = %+v", got.GenFirstMS.Delta)
	}
}

func TestCompareRejectsUnalignedWorkloads(t *testing.T) {
	base := []Row{{CtxTokens: 1, PrefillTokens: 1, GenTokens: 1, GenSteadyTokens: 1}}
	target := []Row{{CtxTokens: 2, PrefillTokens: 1, GenTokens: 1, GenSteadyTokens: 1}}
	if _, err := Compare(base, target); err == nil || !strings.Contains(err.Error(), "workload differs") {
		t.Fatalf("err=%v", err)
	}
}

func TestCompareRejectsInvalidInput(t *testing.T) {
	base := []Row{{CtxTokens: 1, PrefillTokens: 1, GenTokens: 1, GenSteadyTokens: 1, PrefillTPS: math.NaN()}}
	target := base
	if _, err := Compare(base, target); err == nil || !strings.Contains(err.Error(), "base row 0") {
		t.Fatalf("err=%v", err)
	}
	if _, err := Compare(nil, target); err == nil {
		t.Fatal("accepted empty base")
	}
}
