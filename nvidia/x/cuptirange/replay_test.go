package cuptirange

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestReplayRuns(t *testing.T) {
	for _, mode := range []string{"complete", "incomplete", "start", "run", "stop", "invalid flag"} {
		t.Run(mode, func(t *testing.T) {
			failure := errors.New("native or workload failure")
			starts, runs, stops := 0, 0, 0
			err := replayRuns(3,
				func() error {
					starts++
					if mode == "start" {
						return failure
					}
					return nil
				},
				func(pass uint) error {
					if pass != uint(runs) {
						t.Error("wrong pass index")
					}
					runs++
					if mode == "run" {
						return failure
					}
					return nil
				},
				func() (cupti.RangeProfilerStopParams, error) {
					stops++
					state := cupti.RangeProfilerStopParams{}
					if mode == "complete" && stops == 2 {
						state.IsAllPassSubmitted = 1
					}
					if mode == "invalid flag" {
						state.IsAllPassSubmitted = 2
					}
					if mode == "stop" {
						return state, failure
					}
					return state, nil
				})
			if (err == nil) != (mode == "complete") {
				t.Fatalf("error=%v", err)
			}
			if mode == "start" {
				if starts != 1 || runs != 0 || stops != 0 {
					t.Fatal("ran after start failure")
				}
				return
			}
			if starts != stops || runs != stops {
				t.Fatal("unbalanced pass")
			}
			if mode == "incomplete" && runs != 3 {
				t.Fatal("replay limit not enforced")
			}
			if mode == "complete" && runs != 2 {
				t.Fatal("ignored native completion flag")
			}
			if (mode == "run" || mode == "stop") && !errors.Is(err, failure) {
				t.Fatal("lost failure cause")
			}
		})
	}
}

func TestReplayPanic(t *testing.T) {
	stops := 0
	defer func() {
		if got := recover(); got != "workload panic" {
			t.Errorf("panic = %v, want workload panic", got)
		}
		if stops != 1 {
			t.Errorf("Stop calls = %d, want 1", stops)
		}
	}()
	_ = replayRuns(2, func() error { return nil }, func(uint) error {
		panic("workload panic")
	}, func() (cupti.RangeProfilerStopParams, error) {
		stops++
		return cupti.RangeProfilerStopParams{}, nil
	})
}
