package cudaliveness_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

// holding returns an Executed control for which all three conditions hold.
func holding() cudaliveness.Executed {
	return cudaliveness.Executed{
		Site:      "intmix@0x10",
		Predicted: "p1",
		Why:       "S2R R7, SR_TID.Y replaces SR_TID.X: out[i] changes for threads with tid.y != tid.x",
		Unpatched: "u0",
		Patched:   "p1",
		Ran:       true,
	}
}

func TestExecutedCheck(t *testing.T) {
	const mirror = ".nv.capmerc.text.intmix"
	tests := []struct {
		name    string
		edit    func(*cudaliveness.Executed)
		want    error   // nil, ErrUnexplainedPrediction, ErrNotLive or ErrHardwarePending
		notWant []error // sentinels the error must not match
		frag    string  // the message must contain it
		notFrag string  // the message must not contain it
	}{
		{name: "holds", edit: func(*cudaliveness.Executed) {}},

		{name: "no prediction", edit: func(e *cudaliveness.Executed) { e.Predicted = "" },
			want: cudaliveness.ErrUnexplainedPrediction, notWant: []error{cudaliveness.ErrNotLive, cudaerrors.ErrHardwarePending},
			frag: "no predicted output"},
		{name: "no derivation", edit: func(e *cudaliveness.Executed) { e.Why = "" },
			want: cudaliveness.ErrUnexplainedPrediction, notWant: []error{cudaliveness.ErrNotLive, cudaerrors.ErrHardwarePending},
			frag: "predicted p1"},
		{name: "no derivation unrun", edit: func(e *cudaliveness.Executed) { e.Why = ""; e.Ran = false },
			want: cudaliveness.ErrUnexplainedPrediction, notWant: []error{cudaerrors.ErrHardwarePending}},

		{name: "mirrored", edit: func(e *cudaliveness.Executed) { e.Mirrors = []string{mirror} },
			want: cudaliveness.ErrNotLive, notWant: []error{cudaerrors.ErrHardwarePending},
			frag: mirror},
		{name: "mirrored unrun", edit: func(e *cudaliveness.Executed) {
			e.Mirrors = []string{mirror}
			e.Ran, e.Patched, e.Unpatched = false, "", ""
		},
			want: cudaliveness.ErrNotLive, notWant: []error{cudaerrors.ErrHardwarePending},
			frag: mirror},
		{name: "mirrored and inert", edit: func(e *cudaliveness.Executed) { e.Mirrors = []string{mirror}; e.Patched = "u0" },
			want: cudaliveness.ErrNotLive, frag: mirror, notFrag: "equals the unpatched"},
		{name: "empty mirror name", edit: func(e *cudaliveness.Executed) { e.Mirrors = []string{""} },
			want: cudaliveness.ErrNotLive, notWant: []error{cudaerrors.ErrHardwarePending},
			frag: `[""]`},

		{name: "unrun", edit: func(e *cudaliveness.Executed) { e.Ran = false },
			want: cudaerrors.ErrHardwarePending, notWant: []error{cudaliveness.ErrNotLive}},
		{name: "no patched run", edit: func(e *cudaliveness.Executed) { e.Patched = "" },
			want: cudaerrors.ErrHardwarePending, notWant: []error{cudaliveness.ErrNotLive}},
		{name: "no unpatched run", edit: func(e *cudaliveness.Executed) { e.Unpatched = "" },
			want: cudaerrors.ErrHardwarePending, notWant: []error{cudaliveness.ErrNotLive}},
		{name: "both empty", edit: func(e *cudaliveness.Executed) { e.Patched, e.Unpatched = "", "" },
			want: cudaerrors.ErrHardwarePending, notWant: []error{cudaliveness.ErrNotLive}},

		{name: "inert", edit: func(e *cudaliveness.Executed) { e.Patched = "u0" },
			want: cudaliveness.ErrNotLive, frag: "equals the unpatched output"},
		{name: "inert matching prediction", edit: func(e *cudaliveness.Executed) { e.Predicted, e.Patched = "u0", "u0" },
			want: cudaliveness.ErrNotLive, frag: "equals the unpatched output"},
		{name: "mispredicted", edit: func(e *cudaliveness.Executed) { e.Patched = "p2" },
			want: cudaliveness.ErrNotLive, frag: "predicted p1 (S2R R7, SR_TID.Y replaces SR_TID.X"},
		{name: "predicts no change", edit: func(e *cudaliveness.Executed) { e.Predicted = "u0" },
			want: cudaliveness.ErrNotLive, frag: "predicted u0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := holding()
			tt.edit(&e)
			err := e.Check()
			if tt.want == nil {
				if err != nil {
					t.Fatalf("Check() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, tt.want) {
				t.Fatalf("Check() = %v, want match for %v", err, tt.want)
			}
			for _, nw := range tt.notWant {
				if errors.Is(err, nw) {
					t.Errorf("Check() = %v, must not match %v", err, nw)
				}
			}
			msg := err.Error()
			if !strings.HasPrefix(msg, e.Name()+": ") {
				t.Errorf("Check() = %q, want it to start with the control name %q", msg, e.Name())
			}
			if tt.frag != "" && !strings.Contains(msg, tt.frag) {
				t.Errorf("Check() = %q, want it to contain %q", msg, tt.frag)
			}
			if tt.notFrag != "" && strings.Contains(msg, tt.notFrag) {
				t.Errorf("Check() = %q, must not contain %q", msg, tt.notFrag)
			}
		})
	}
}

// Check reads Mirrors and does not modify it.
func TestExecutedLeavesMirrorsAlone(t *testing.T) {
	mirrors := []string{".nv.capmerc.text.b", ".nv.capmerc.text.a"}
	e := holding()
	e.Mirrors = mirrors
	err := e.Check()
	if mirrors[0] != ".nv.capmerc.text.b" || mirrors[1] != ".nv.capmerc.text.a" {
		t.Errorf("Check modified Mirrors: %q", mirrors)
	}
	if want := `[".nv.capmerc.text.b" ".nv.capmerc.text.a"]`; err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("Check() = %v, want the mirrors in the order given, %s", err, want)
	}
}

// A patch nobody launched is not a patch that failed to execute.
func TestExecutedUnrunIsNotAFailure(t *testing.T) {
	unrun := cudaliveness.Assert(cudaliveness.Executed{
		Site:      "intmix@0x10",
		Predicted: "p1",
		Why:       "S2R R7, SR_TID.Y replaces SR_TID.X",
	})
	if errors.Is(unrun, cudaliveness.ErrNotLive) {
		t.Fatal("an unrun Executed control reported ErrNotLive; absence was encoded as a negative")
	}
	if !errors.Is(unrun, cudaerrors.ErrHardwarePending) {
		t.Fatalf("Assert() = %v, want ErrHardwarePending", unrun)
	}
}

func TestExecutedAssertJoins(t *testing.T) {
	mirrored := holding()
	mirrored.Mirrors = []string{".nv.capmerc.text.intmix"}
	err := cudaliveness.Assert(
		mirrored,                             // failed
		cudaliveness.Planted{Site: "intmix"}, // unrun
		cudaliveness.Counted{Site: "intmix", Want: 2, Got: 2, Why: "one read, one write", Ran: true},
	)
	if !errors.Is(err, cudaliveness.ErrNotLive) {
		t.Errorf("Assert() = %v, want it to match ErrNotLive", err)
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("Assert() = %v, want it to match ErrHardwarePending", err)
	}
	if err == nil {
		t.Fatal("Assert() = nil")
	}
	lines := strings.Split(err.Error(), "\n")
	if len(lines) != 2 {
		t.Fatalf("Assert() reported %d failures, want 2:\n%v", len(lines), err)
	}
	if !strings.Contains(err.Error(), ".nv.capmerc.text.intmix") {
		t.Errorf("Assert() = %v, want a line naming the mirror", err)
	}

	if err := cudaliveness.Assert(
		holding(),
		cudaliveness.Undisturbed{Site: "out", ObserverOff: "u0", ObserverOn: "u0", Ran: true},
	); err != nil {
		t.Errorf("Assert(holding Executed, holding Undisturbed) = %v, want nil", err)
	}
}
