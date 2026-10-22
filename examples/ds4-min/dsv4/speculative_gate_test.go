package dsv4

import (
	"math"
	"testing"
)

func TestCheckSpeculativeIdentityPassesBeforeNearTie(t *testing.T) {
	report, err := CheckSpeculativeIdentity([]SpeculativeTrace{{
		Strict:         []int{4, 7, 9},
		Speculative:    []int{4, 7, 3},
		Margins:        []float64{0.4, 0.2, 0.0005},
		AcceptedDrafts: 2,
		Cycles:         1,
	}}, 1e-3)
	if err != nil || !report.Pass() {
		t.Fatalf("report=%+v err=%v", report, err)
	}
	if report.EligiblePositions != 2 || report.ToleratedMismatches != 1 || report.Mismatches != 0 {
		t.Fatalf("report=%+v", report)
	}
}

func TestCheckSpeculativeIdentityMutationFails(t *testing.T) {
	report, err := CheckSpeculativeIdentity([]SpeculativeTrace{{
		Strict:         []int{4, 7},
		Speculative:    []int{5, 7},
		Margins:        []float64{0.4, 0.2},
		AcceptedDrafts: 1,
		Cycles:         1,
	}}, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	if report.Pass() || report.Mismatches != 1 || report.FirstMismatchPosition != 0 {
		t.Fatalf("mutation report=%+v", report)
	}
}

func TestCheckSpeculativeIdentityKeepsFirstNearTie(t *testing.T) {
	report, err := CheckSpeculativeIdentity([]SpeculativeTrace{{
		Strict:         []int{4, 7, 9, 2},
		Speculative:    []int{4, 3, 5, 6},
		Margins:        []float64{0.4, 0.0005, 0.2, 0.0001},
		AcceptedDrafts: 1,
		Cycles:         1,
	}}, 1e-3)
	if err != nil || !report.Pass() || report.EligiblePositions != 1 || report.ToleratedMismatches != 3 {
		t.Fatalf("report=%+v err=%v", report, err)
	}
}

func TestCheckSpeculativeIdentityRejectsVacuousOrMalformedTrace(t *testing.T) {
	tests := []struct {
		name   string
		traces []SpeculativeTrace
		margin float64
	}{
		{"empty", nil, 1e-3},
		{"no accepted drafts", []SpeculativeTrace{{Strict: []int{1}, Speculative: []int{1}, Margins: []float64{1}, Cycles: 1}}, 1e-3},
		{"length mismatch", []SpeculativeTrace{{Strict: []int{1}, Speculative: []int{1, 2}, Margins: []float64{1}, AcceptedDrafts: 1, Cycles: 1}}, 1e-3},
		{"invalid margin", []SpeculativeTrace{{Strict: []int{1}, Speculative: []int{1}, Margins: []float64{math.NaN()}, AcceptedDrafts: 1, Cycles: 1}}, 1e-3},
		{"nan after near tie", []SpeculativeTrace{{Strict: []int{1, 2, 3}, Speculative: []int{1, 2, 3}, Margins: []float64{1, 0, math.NaN()}, AcceptedDrafts: 1, Cycles: 1}}, 1e-3},
		{"infinity after near tie", []SpeculativeTrace{{Strict: []int{1, 2, 3}, Speculative: []int{1, 2, 3}, Margins: []float64{1, 0, math.Inf(1)}, AcceptedDrafts: 1, Cycles: 1}}, 1e-3},
		{"negative after near tie", []SpeculativeTrace{{Strict: []int{1, 2, 3}, Speculative: []int{1, 2, 3}, Margins: []float64{1, 0, -1}, AcceptedDrafts: 1, Cycles: 1}}, 1e-3},
		{"invalid final case", []SpeculativeTrace{
			{Strict: []int{1}, Speculative: []int{1}, Margins: []float64{1}, AcceptedDrafts: 1, Cycles: 1},
			{Strict: []int{1}, Speculative: []int{1}, Margins: []float64{math.NaN()}, AcceptedDrafts: 1, Cycles: 1},
		}, 1e-3},
		{"all near ties", []SpeculativeTrace{{Strict: []int{1}, Speculative: []int{1}, Margins: []float64{0}, AcceptedDrafts: 1, Cycles: 1}}, 1e-3},
		{"invalid threshold", []SpeculativeTrace{{Strict: []int{1}, Speculative: []int{1}, Margins: []float64{1}, AcceptedDrafts: 1, Cycles: 1}}, math.Inf(1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			report, err := CheckSpeculativeIdentity(test.traces, test.margin)
			if err == nil || report.Pass() {
				t.Fatalf("report=%+v err=%v, want fail-closed error", report, err)
			}
		})
	}
}
