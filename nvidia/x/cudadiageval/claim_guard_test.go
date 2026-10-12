package cudadiageval

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestReportOnlyComparativeClaimRefused(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*EvaluationReport)
	}{
		{"complete asserted", func(*EvaluationReport) {}},
		{"empty gates", func(r *EvaluationReport) { r.Gates = nil }},
		{"invented baseline", func(r *EvaluationReport) {
			r.Baselines = []Baseline{{Name: "invented", Version: "unverified", ComparisonScope: "claimed"}}
		}},
		{"historical omitted withholding", func(r *EvaluationReport) {
			r.Diagnostic = DiagnosticReceipt{}
			if err := json.Unmarshal([]byte(`{"no_fault_total":4,"no_fault_passed":4,"no_fault_missing":0}`), &r.Diagnostic); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := noFaultClaimReport()
			tc.mutate(r)
			raw, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			var imported EvaluationReport
			if err := json.Unmarshal(raw, &imported); err != nil {
				t.Fatal(err)
			}
			if err := validateNoFaultClaimReceipt(&imported); err != nil {
				t.Fatalf("local no-fault precondition: %v", err)
			}
			claim, err := FormulateClaim(&imported)
			if claim != nil || !errors.Is(err, ErrIncompleteClaim) || !strings.Contains(err.Error(), "paired candidate and comparator evidence required") {
				t.Fatalf("report-only bypass: claim=%+v err=%v", claim, err)
			}
		})
	}
}

func TestReportOnlyNoFaultRefusalPreserved(t *testing.T) {
	r := noFaultClaimReport()
	r.Diagnostic.NoFaultPassed = 3
	r.Diagnostic.NoFaultWithheld = 1
	r.Protocol.Budgets.MinNoFaultRejectionPct = 75
	_, err := FormulateClaim(r)
	if err == nil || !strings.Contains(err.Error(), "no-fault qualification incomplete") || !strings.Contains(err.Error(), "1 diagnoses withheld") {
		t.Fatalf("no-fault guard bypassed: %v", err)
	}
	r.Diagnostic.NoFaultPassed = 4
	r.Diagnostic.NoFaultWithheld = -1
	_, err = FormulateClaim(r)
	if err == nil || !strings.Contains(err.Error(), "invalid no-fault partition") {
		t.Fatalf("malformed no-fault guard lost: %v", err)
	}
}
