package cuptibench

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func TestConfigPassBudget(t *testing.T) {
	for _, tc := range []struct {
		name          string
		mode          Mode
		replay        cuptirange.ReplayMode
		passes, limit uint
		wantErr       bool
	}{
		{"kernel internal passes", Kernels, cuptirange.KernelReplay, 4, 1, false},
		{"kernel user replay insufficient", Kernels, cuptirange.UserReplay, 4, 1, true},
		{"whole insufficient", Whole, cuptirange.UserReplay, 4, 1, true},
		{"regions insufficient", Regions, cuptirange.UserReplay, 4, 1, true},
		{"whole exact", Whole, cuptirange.UserReplay, 4, 4, false},
		// The host API does not expose the exact nesting schedule. Admission
		// is not a promise of completion; Capture enforces the host bound.
		{"regions lower bound", Regions, cuptirange.UserReplay, 1, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := Config{Mode: tc.mode, ReplayMode: tc.replay, Limits: cuptirange.Limits{MaxPasses: tc.limit}}
			if err := c.checkPassBudget(tc.passes); (err != nil) != tc.wantErr {
				t.Fatalf("checkPassBudget(%d): %v, want error %v", tc.passes, err, tc.wantErr)
			}
		})
	}
}
