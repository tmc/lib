package cudadiageval

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func TestNoFaultPresentWithholdingIsNotCorrect(t *testing.T) {
	res := diagnosticFixture(t)
	for _, id := range []string{"n0", "n1", "n2", "n3"} {
		res.DiagnosticInputs[id] = cudadiag.DiagnosticInput{}
	}
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	g := findGate(t, gates, "no_fault_rejection")
	if r.NoFaultTotal != 4 || r.NoFaultPassed != 0 || r.NoFaultMissing != 0 || g.Disposition != "withheld" {
		t.Fatalf("present withholding counted as correctness: %d/%d missing=%d gate=%+v", r.NoFaultPassed, r.NoFaultTotal, r.NoFaultMissing, g)
	}
	if !strings.Contains(g.Reason, "specificity undefined") || !strings.Contains(g.Reason, "false alarms 0/0 (undefined)") {
		t.Errorf("no admitted answers have undefined specificity: %s", g.Reason)
	}
}

func TestNoFaultPartition(t *testing.T) {
	for _, tc := range []struct {
		name                              string
		mutate                            func(*testing.T, *FixtureRunResult)
		passed, withheld, missing, alarms int
		want                              string
	}{
		{"normal", func(*testing.T, *FixtureRunResult) {}, 4, 0, 0, 0, "passed"},
		{"empty", func(_ *testing.T, r *FixtureRunResult) { r.DiagnosticInputs["n0"] = cudadiag.DiagnosticInput{} }, 3, 1, 0, 0, "withheld"},
		{"loss", func(t *testing.T, r *FixtureRunResult) { withhold(t, r, "n0") }, 3, 1, 0, 0, "withheld"},
		{"missing", func(_ *testing.T, r *FixtureRunResult) { delete(r.DiagnosticInputs, "n0") }, 3, 0, 1, 0, "withheld"},
		{"host alarm", func(_ *testing.T, r *FixtureRunResult) { r.DiagnosticInputs["n0"] = r.DiagnosticInputs["h0"] }, 3, 0, 0, 1, "failed"},
		{"device alarm", func(_ *testing.T, r *FixtureRunResult) { r.DiagnosticInputs["n0"] = r.DiagnosticInputs["d0"] }, 3, 0, 0, 1, "failed"},
		{"ambiguous alarm", func(t *testing.T, r *FixtureRunResult) {
			f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionAmbiguousSimultaneous)
			d := cudadiag.Diagnose(f.Workload, f.OpID, f.Input)
			if !d.Admitted || d.Mechanism != cudadiag.MechanismAmbiguousSimultaneous {
				t.Fatalf("ambiguous control is not admitted: %+v", d)
			}
			for i := range r.GroundTruth {
				if r.GroundTruth[i].RequestID == "n0" {
					r.GroundTruth[i].ProcessGen = f.Workload.Process
					r.GroundTruth[i].WorkloadName = f.Workload.WorkloadName
					r.GroundTruth[i].CorrelationID = uint32(f.OpID)
				}
			}
			r.DiagnosticInputs["n0"] = f.Input
		}, 3, 0, 0, 1, "failed"},
		{"mixed", func(t *testing.T, r *FixtureRunResult) {
			withhold(t, r, "n0")
			delete(r.DiagnosticInputs, "n1")
			r.DiagnosticInputs["n2"] = r.DiagnosticInputs["h0"]
		}, 1, 1, 1, 1, "failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res := diagnosticFixture(t)
			tc.mutate(t, res)
			// Verify the public diagnostic result, independently of the evaluator counts.
			for _, gt := range res.GroundTruth {
				if gt.Fault != FaultNone {
					continue
				}
				if in, ok := res.DiagnosticInputs[gt.RequestID]; ok {
					d := cudadiag.Diagnose(cudaworkload.WorkloadIdentity{Process: gt.ProcessGen, WorkloadName: gt.WorkloadName}, cudaworkload.OpID(gt.CorrelationID), in)
					if (tc.name == "empty" || tc.name == "loss") && gt.RequestID == "n0" && d.Admitted {
						t.Fatalf("withholding control admitted: %+v", d)
					}
				}
			}
			r, gates := evaluateDiagnostics(res, DefaultBudgets())
			alarms := r.NoFaultTotal - r.NoFaultPassed - r.NoFaultWithheld - r.NoFaultMissing
			if r.NoFaultTotal != 4 || r.NoFaultPassed != tc.passed || r.NoFaultWithheld != tc.withheld || r.NoFaultMissing != tc.missing || alarms != tc.alarms {
				t.Fatalf("partition total=%d passed=%d withheld=%d missing=%d alarms=%d, want 4 %d %d %d %d", r.NoFaultTotal, r.NoFaultPassed, r.NoFaultWithheld, r.NoFaultMissing, alarms, tc.passed, tc.withheld, tc.missing, tc.alarms)
			}
			g := findGate(t, gates, "no_fault_rejection")
			if g.Disposition != tc.want {
				t.Errorf("gate=%+v, want %s", g, tc.want)
			}
			if r.GateDisposition != tc.want {
				t.Errorf("receipt disposition=%s, want %s", r.GateDisposition, tc.want)
			}
		})
	}
}

func TestNoFaultGateCoverage(t *testing.T) {
	for _, tc := range []struct {
		name                             string
		passed, withheld, missing, total int
		min                              float64
		want                             string
	}{
		{"complete", 4, 0, 0, 4, 100, "passed"},
		{"relaxed withheld", 3, 1, 0, 4, 75, "withheld"},
		{"relaxed missing", 3, 0, 1, 4, 75, "withheld"},
		{"optimistic cannot reach", 1, 1, 1, 4, 100, "failed"},
		{"relaxed false alarms", 3, 0, 0, 4, 75, "passed"},
		{"below relaxed", 2, 0, 0, 4, 75, "failed"},
		{"no cases", 0, 0, 0, 0, 100, "failed"},
		{"all unanswered", 0, 2, 2, 4, 100, "withheld"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, reason := noFaultGate(tc.passed, tc.withheld, tc.missing, tc.total, tc.min)
			if d != tc.want {
				t.Fatalf("%s: %s, want %s", d, reason, tc.want)
			}
			if tc.total > 0 && (!strings.Contains(reason, "diagnoses withheld") || !strings.Contains(reason, "inputs missing") || !strings.Contains(reason, "admitted coverage") || !strings.Contains(reason, "false alarms")) {
				t.Errorf("incomplete reason: %s", reason)
			}
		})
	}
}

func TestNoFaultGateLargeCounts(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                             string
		passed, withheld, missing, total int
		min                              float64
		want                             string
	}{
		{"max complete", maxInt, 0, 0, maxInt, 100, "passed"},
		{"max complete relaxed", maxInt, 0, 0, maxInt, 75, "passed"},
		{"max single alarm", maxInt - 1, 0, 0, maxInt, 100, "failed"},
		{"max single withheld", maxInt - 1, 1, 0, maxInt, 100, "withheld"},
		{"max single missing", maxInt - 1, 0, 1, maxInt, 100, "withheld"},
		{"bounded sum overflow", maxInt, maxInt, 0, maxInt, 100, "failed"},
		{"bounded missing overflow", maxInt, 0, maxInt, maxInt, 100, "failed"},
		{"bounded sum relaxed overflow", maxInt, maxInt, maxInt, maxInt, 75, "failed"},
		{"negative pass", -1, 0, 0, 4, 100, "failed"},
		{"negative withheld", 4, -1, 0, 4, 100, "failed"},
		{"negative missing", 4, 0, -1, 4, 100, "failed"},
		{"negative total", 0, 0, 0, -1, 100, "failed"},
		{"nan threshold", 4, 0, 0, 4, math.NaN(), "failed"},
		{"infinite threshold", 4, 0, 0, 4, math.Inf(1), "failed"},
		{"zero threshold", 4, 0, 0, 4, 0, "failed"},
		{"oversized threshold", 4, 0, 0, 4, 101, "failed"},
		{"exact half threshold", maxInt / 2, 0, 0, maxInt, 50, "failed"},
		{"above half threshold", maxInt/2 + 1, 0, 0, maxInt, 50, "passed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, reason := noFaultGate(tc.passed, tc.withheld, tc.missing, tc.total, tc.min)
			if d != tc.want {
				t.Errorf("%s: %s, want %s", d, reason, tc.want)
			}
		})
	}
}

func TestNoFaultWithheldJSON(t *testing.T) {
	b, err := json.Marshal(DiagnosticReceipt{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"no_fault_withheld":0`) || !strings.Contains(string(b), `"no_fault_missing":0`) {
		t.Fatalf("zero counts omitted: %s", b)
	}
	in := DiagnosticReceipt{NoFaultTotal: 4, NoFaultPassed: 1, NoFaultWithheld: 2, NoFaultMissing: 1}
	b, err = json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out DiagnosticReceipt
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.NoFaultTotal != 4 || out.NoFaultPassed != 1 || out.NoFaultWithheld != 2 || out.NoFaultMissing != 1 {
		t.Fatalf("round trip changed partition: %+v", out)
	}
}

// noFaultClaimReport is an imported report with summary assertions. It
// exercises no-fault admission only and is not authentic qualification evidence.
func noFaultClaimReport() *EvaluationReport {
	return &EvaluationReport{
		Protocol:        Protocol{Name: "no-fault-control", Phase: PhaseQualification, Frozen: true, Budgets: DefaultBudgets()},
		Passed:          true,
		Gates:           []PropertyGate{{GateName: "no_fault_rejection", Disposition: "passed"}},
		Diagnostic:      DiagnosticReceipt{NoFaultTotal: 4, NoFaultPassed: 4},
		Baselines:       DefaultBaselines(),
		Reproducibility: ReproducibilityReceipt{ArtifactManifestSHA256: strings.Repeat("a", 64)},
	}
}

func TestNoFaultClaimAdmission(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                             string
		passed, withheld, missing, total int
		min                              float64
		want                             bool
	}{
		{"complete", 4, 0, 0, 4, 100, true},
		{"present withheld", 3, 1, 0, 4, 75, false},
		{"missing", 3, 0, 1, 4, 75, false},
		{"false alarm", 3, 0, 0, 4, 100, false},
		{"relaxed complete", 3, 0, 0, 4, 75, true},
		{"negative pass", -1, 0, 0, 4, 100, false},
		{"negative withheld", 4, -1, 0, 4, 100, false},
		{"negative missing", 4, 0, -1, 4, 100, false},
		{"negative total", 0, 0, 0, -1, 100, false},
		{"no cases", 0, 0, 0, 0, 100, false},
		{"sum too large", 4, 1, 0, 4, 100, false},
		{"integer overflow", maxInt, maxInt, maxInt, maxInt, 100, false},
		{"single alarm at max", maxInt - 1, 0, 0, maxInt, 100, false},
		{"nan threshold", 4, 0, 0, 4, math.NaN(), false},
		{"infinite threshold", 4, 0, 0, 4, math.Inf(1), false},
		{"zero threshold", 4, 0, 0, 4, 0, false},
		{"too high threshold", 4, 0, 0, 4, 101, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := noFaultClaimReport()
			r.Diagnostic = DiagnosticReceipt{NoFaultPassed: tc.passed, NoFaultWithheld: tc.withheld, NoFaultMissing: tc.missing, NoFaultTotal: tc.total}
			r.Protocol.Budgets.MinNoFaultRejectionPct = tc.min
			err := validateNoFaultClaimReceipt(r)
			if (err == nil) != tc.want {
				t.Fatalf("no-fault error=%v, want accepted=%t", err, tc.want)
			}
			c, claimErr := FormulateClaim(r)
			if claimErr == nil || c != nil {
				t.Fatalf("report-only comparative claim accepted: %+v", c)
			}
			if tc.want && !strings.Contains(claimErr.Error(), "paired candidate and comparator evidence required") {
				t.Fatalf("complete no-fault receipt has wrong comparative refusal: %v", claimErr)
			}
		})
	}
}

func TestNoFaultClaimImportedJSON(t *testing.T) {
	r := noFaultClaimReport()
	// Both the summary Passed flag and gate say passed, but the receipt
	// carries unanswered diagnoses. JSON import must preserve that refusal.
	if err := json.Unmarshal([]byte(`{"no_fault_total":4,"no_fault_passed":3,"no_fault_withheld":1,"no_fault_missing":0}`), &r.Diagnostic); err != nil {
		t.Fatal(err)
	}
	r.Protocol.Budgets.MinNoFaultRejectionPct = 75
	if c, err := FormulateClaim(r); err == nil {
		t.Fatalf("imported withheld receipt bypassed admission: %+v", c)
	}
}

func TestNoFaultOverflowStaysResilience(t *testing.T) {
	res := resilienceRun(t, ResilienceOverflow)
	r, gates := evaluateDiagnostics(res, DefaultBudgets())
	if r.NoFaultTotal != 2 || r.NoFaultPassed != 2 || r.NoFaultWithheld != 0 || r.NoFaultMissing != 0 || r.ResilienceTotal != 1 || r.ResiliencePassed != 1 {
		t.Fatalf("overflow changed no-fault partition: %+v", r)
	}
	if g := findGate(t, gates, "system_resilience_controls"); g.Disposition != "passed" {
		t.Fatalf("expected overflow refusal no longer passes resilience: %+v", g)
	}
}

func TestNoFaultRelaxedCoverage(t *testing.T) {
	for _, missing := range []bool{false, true} {
		res := diagnosticFixture(t)
		if missing {
			delete(res.DiagnosticInputs, "n0")
		} else {
			withhold(t, res, "n0")
		}
		budgets := DefaultBudgets()
		budgets.MinNoFaultRejectionPct = 75
		r, gates := evaluateDiagnostics(res, budgets)
		g := findGate(t, gates, "no_fault_rejection")
		if r.NoFaultPassed != 3 || r.NoFaultTotal != 4 || g.MeasuredValue != "75.0%" || g.Disposition != "withheld" || r.GateDisposition != "withheld" {
			t.Fatalf("75%% correctness with incomplete coverage was accepted: receipt=%+v gate=%+v", r, g)
		}
		if !strings.Contains(g.Reason, "admitted correctness 3/4") || !strings.Contains(g.Reason, "admitted coverage 3/4") || !strings.Contains(g.Reason, "false alarms 0/3") {
			t.Errorf("denominators omitted: %s", g.Reason)
		}
	}
}
