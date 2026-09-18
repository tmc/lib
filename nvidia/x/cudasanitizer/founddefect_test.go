package cudasanitizer

import "testing"

// TestFoundDefectSeparatesNoVerdictFromDefect is the regression for the failure
// this predicate exists to prevent: a timeout must not read as a caught defect.
func TestFoundDefectSeparatesNoVerdictFromDefect(t *testing.T) {
	tests := []struct {
		name string
		res  Result
		want bool
	}{
		{"timeout, no verdict", Result{Outcome: OutcomeToolFailure, Err: ErrTimeout}, false},
		{"tool missing", Result{Outcome: OutcomeToolFailure, Err: ErrToolNotFound}, false},
		{"clean", Result{Outcome: OutcomeClean}, false},
		{"application failed", Result{Outcome: OutcomeApplicationFailed}, false},
		{"errors reported", Result{Outcome: OutcomeSanitizerErrors, Log: &Log{Errors: 3}}, true},
		{"races reported", Result{Outcome: OutcomeSanitizerErrors, Log: &Log{RaceErrors: 1}}, true},
		{"hazards reported", Result{Outcome: OutcomeSanitizerErrors, Log: &Log{Hazards: 2}}, true},
		{"errors outcome, zero counts", Result{Outcome: OutcomeSanitizerErrors, Log: &Log{}}, false},
		{"errors outcome, no log at all", Result{Outcome: OutcomeSanitizerErrors}, false},
	}
	for _, tt := range tests {
		if got := tt.res.FoundDefect(); got != tt.want {
			t.Errorf("%s: FoundDefect() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// TestNotCleanIsNotFoundDefect states the trap in the form a caller meets it:
// the two disagree exactly on the no-verdict outcomes, which is the point.
func TestNotCleanIsNotFoundDefect(t *testing.T) {
	r := Result{Outcome: OutcomeToolFailure, Err: ErrTimeout}
	if r.Outcome == OutcomeClean {
		t.Fatal("fixture is wrong: a timeout must not be clean")
	}
	if r.FoundDefect() {
		t.Error("a timed-out run reports a defect; the !clean shorthand is back")
	}
}
