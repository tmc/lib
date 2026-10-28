package cudaliveness_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

func TestExecutedUnrunPrecedence(t *testing.T) {
	runs := []struct {
		name      string
		ran       bool
		patched   string
		unpatched string
	}{
		{"not launched", false, "p1", "u0"},
		{"not launched without patched", false, "", "u0"},
		{"not launched without unpatched", false, "p1", ""},
		{"not launched without digests", false, "", ""},
		{"missing patched", true, "", "u0"},
		{"missing unpatched", true, "p1", ""},
		{"missing both", true, "", ""},
	}
	predictions := []struct {
		name      string
		predicted string
		why       string
	}{
		{"derived", "p1", "derived before launch"},
		{"no prediction", "", "derived before launch"},
		{"no derivation", "p1", ""},
		{"neither", "", ""},
	}
	mirrors := []struct {
		name  string
		names []string
	}{
		{"no mirrors", nil},
		{"mirror", []string{".nv.capmerc.text.intmix"}},
		{"empty mirror name", []string{""}},
	}
	for _, run := range runs {
		for _, prediction := range predictions {
			for _, mirror := range mirrors {
				t.Run(run.name+"/"+prediction.name+"/"+mirror.name, func(t *testing.T) {
					e := cudaliveness.Executed{
						Site: "intmix@0x10", Ran: run.ran,
						Patched: run.patched, Unpatched: run.unpatched,
						Predicted: prediction.predicted, Why: prediction.why, Mirrors: mirror.names,
					}
					for _, check := range []struct {
						name string
						err  error
					}{
						{"Check", e.Check()},
						{"Assert", cudaliveness.Assert(e)},
					} {
						if !errors.Is(check.err, cudaerrors.ErrHardwarePending) {
							t.Errorf("%s = %v, want ErrHardwarePending", check.name, check.err)
						}
						if errors.Is(check.err, cudaliveness.ErrNotLive) || errors.Is(check.err, cudaliveness.ErrUnexplainedPrediction) {
							t.Errorf("%s = %v, unrun control must not report a failed control", check.name, check.err)
						}
					}
				})
			}
		}
	}
	t.Run("zero value", func(t *testing.T) {
		if err := (cudaliveness.Executed{}).Check(); !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("Check = %v, want ErrHardwarePending", err)
		}
	})
}
