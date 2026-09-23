package cudaliveness_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

func TestControlCheck(t *testing.T) {
	tests := []struct {
		name    string
		control cudaliveness.Control
		want    error // nil, cudaerrors.ErrHardwarePending, ErrNotLive, or ErrUnexplainedCount
	}{
		{"planted found", cudaliveness.Planted{Site: "k", Reported: 1, Ran: true}, nil},
		{"planted missed", cudaliveness.Planted{Site: "k", Reported: 0, Ran: true}, cudaliveness.ErrNotLive},
		{"planted unrun", cudaliveness.Planted{Site: "k", Reported: 1}, cudaerrors.ErrHardwarePending},

		{"counted exact", cudaliveness.Counted{Site: "k", Want: 2, Got: 2, Why: "one read, one write", Ran: true}, nil},
		{"counted low", cudaliveness.Counted{Site: "k", Want: 2, Got: 0, Why: "one read, one write", Ran: true}, cudaliveness.ErrNotLive},
		{"counted high", cudaliveness.Counted{Site: "k", Want: 2, Got: 9, Why: "one read, one write", Ran: true}, cudaliveness.ErrNotLive},
		{"counted unexplained", cudaliveness.Counted{Site: "k", Want: 2, Got: 2, Ran: true}, cudaliveness.ErrUnexplainedCount},
		{"counted unrun", cudaliveness.Counted{Site: "k", Want: 2, Got: 2, Why: "d"}, cudaerrors.ErrHardwarePending},

		{"refused", cudaliveness.Refused{Site: "sm_90 object", Rejected: true, Ran: true}, nil},
		{"accepted", cudaliveness.Refused{Site: "sm_90 object", Rejected: false, Ran: true}, cudaliveness.ErrNotLive},
		{"refusal unrun", cudaliveness.Refused{Site: "sm_90 object", Rejected: true}, cudaerrors.ErrHardwarePending},

		{"adjacent silent", cudaliveness.Adjacent{Site: "k+1", Reported: 0, Ran: true}, nil},
		{"adjacent reported", cudaliveness.Adjacent{Site: "k+1", Reported: 1, Ran: true}, cudaliveness.ErrNotLive},
		{"adjacent unrun", cudaliveness.Adjacent{Site: "k+1", Reported: 0}, cudaerrors.ErrHardwarePending},

		{"undisturbed", cudaliveness.Undisturbed{Site: "logits", ObserverOff: "ab", ObserverOn: "ab", Ran: true}, nil},
		{"disturbed", cudaliveness.Undisturbed{Site: "logits", ObserverOff: "ab", ObserverOn: "cd", Ran: true}, cudaliveness.ErrNotLive},
		{"undisturbed unrun", cudaliveness.Undisturbed{Site: "logits", ObserverOff: "ab", ObserverOn: "ab"}, cudaerrors.ErrHardwarePending},
		{"undisturbed empty digest", cudaliveness.Undisturbed{Site: "logits", ObserverOff: "ab", Ran: true}, cudaerrors.ErrHardwarePending},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.control.Check()
			if tt.want == nil {
				if err != nil {
					t.Fatalf("Check() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, tt.want) {
				t.Fatalf("Check() = %v, want match for %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.control.Name()) {
				t.Errorf("Check() = %q, want it to name the control %q", err, tt.control.Name())
			}
		})
	}
}

// A clean report with no control is the failure this package exists to catch,
// so Assert must refuse it rather than agreeing that nothing went wrong.
func TestAssertRefusesNoControls(t *testing.T) {
	if err := Assert(); !errors.Is(err, cudaliveness.ErrNoControls) {
		t.Fatalf("Assert() = %v, want ErrNoControls", err)
	}
}

func Assert(c ...cudaliveness.Control) error { return cudaliveness.Assert(c...) }

func TestAssertHolds(t *testing.T) {
	err := cudaliveness.Assert(
		cudaliveness.Planted{Site: "gemm", Reported: 1, Ran: true},
		cudaliveness.Adjacent{Site: "gemm+1", Reported: 0, Ran: true},
		cudaliveness.Counted{Site: "gemm", Want: 2, Got: 2, Why: "warm-up read and write", Ran: true},
		cudaliveness.Refused{Site: "sm_90 patch", Rejected: true, Ran: true},
		cudaliveness.Undisturbed{Site: "logits", ObserverOff: "d0", ObserverOn: "d0", Ran: true},
	)
	if err != nil {
		t.Fatalf("Assert() = %v, want nil", err)
	}
}

// Assert must report every failure, because an unrun control and a failed one
// call for different work: one is a run nobody made, the other a probe that
// did not observe its target.
func TestAssertJoinsEveryFailure(t *testing.T) {
	err := cudaliveness.Assert(
		cudaliveness.Planted{Site: "gemm", Reported: 0, Ran: true}, // failed
		cudaliveness.Refused{Site: "sm_90 patch", Rejected: true},  // unrun
		cudaliveness.Adjacent{Site: "gemm+1", Reported: 0, Ran: true},
	)
	if !errors.Is(err, cudaliveness.ErrNotLive) {
		t.Errorf("Assert() = %v, want it to match ErrNotLive", err)
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("Assert() = %v, want it to match ErrHardwarePending", err)
	}
	if got := strings.Count(err.Error(), "\n") + 1; got != 2 {
		t.Errorf("Assert() reported %d failures, want 2:\n%v", got, err)
	}
}

// The distinction ErrHardwarePending exists to protect: a probe nobody ran is
// not a probe that found nothing.
func TestUnrunIsNotAFailure(t *testing.T) {
	unrun := cudaliveness.Assert(cudaliveness.Planted{Site: "k"})
	if errors.Is(unrun, cudaliveness.ErrNotLive) {
		t.Fatal("an unrun control reported ErrNotLive; absence was encoded as a negative")
	}
	if !errors.Is(unrun, cudaerrors.ErrHardwarePending) {
		t.Fatalf("Assert() = %v, want ErrHardwarePending", unrun)
	}
}

// Planted alone passes for a probe watching everything. This is why the
// package documents the two as only meaningful together.
func TestPlantedAloneDoesNotLocalize(t *testing.T) {
	wideOpen := cudaliveness.Planted{Site: "gemm", Reported: 1, Ran: true}
	if err := cudaliveness.Assert(wideOpen); err != nil {
		t.Fatalf("Assert() = %v, want nil", err)
	}
	withNegative := cudaliveness.Assert(
		wideOpen,
		cudaliveness.Adjacent{Site: "gemm+1", Reported: 1, Ran: true},
	)
	if !errors.Is(withNegative, cudaliveness.ErrNotLive) {
		t.Fatalf("Assert() = %v, want ErrNotLive: the probe reported a site outside its window", withNegative)
	}
}
