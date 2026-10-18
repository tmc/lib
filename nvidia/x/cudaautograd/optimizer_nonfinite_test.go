package cudaautograd_test

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
)

func TestSGDOptimizerStepRejectsNonFiniteGradients(t *testing.T) {
	values := []struct {
		name  string
		value float32
	}{
		{"nan", float32(math.NaN())},
		{"positive infinity", float32(math.Inf(1))},
		{"negative infinity", float32(math.Inf(-1))},
	}
	for _, value := range values {
		for _, momentum := range []float32{0, 0.5} {
			for _, warm := range []bool{false, true} {
				for badIndex := range 3 {
					name := fmt.Sprintf("%s/momentum=%g/warm=%t/index=%d", value.name, momentum, warm, badIndex)
					t.Run(name, func(t *testing.T) {
						// Repeat with fresh maps so validation cannot depend on
						// encountering the invalid parameter first.
						for range 32 {
							opt, err := cudaautograd.NewSGDOptimizer(0.25, momentum)
							if err != nil {
								t.Fatal(err)
							}
							ref, err := cudaautograd.NewSGDOptimizer(0.25, momentum)
							if err != nil {
								t.Fatal(err)
							}
							params := map[string][]float32{"bad": {4, -2, 1}, "a": {5, 6}, "b": {-7}, "frozen": {8}}
							want := cloneParams(params)
							good := map[string][]float32{"bad": {1, -2, 3}, "a": {2, 4}, "b": {-1}}
							if warm {
								if err := opt.Step(params, good); err != nil {
									t.Fatal(err)
								}
								if err := ref.Step(want, good); err != nil {
									t.Fatal(err)
								}
							}
							before := cloneParams(params)
							bad := cloneParams(good)
							bad["bad"][badIndex] = value.value
							gradsBefore := cloneParams(bad)
							err = opt.Step(params, bad)
							if err == nil || !strings.Contains(err.Error(), `param "bad"`) || !strings.Contains(err.Error(), fmt.Sprintf("gradient[%d]", badIndex)) {
								t.Fatalf("Step = %v, want error naming parameter and gradient index", err)
							}
							if !sameBits(params, before) {
								t.Fatalf("refused Step changed parameters: got %v, want %v", params, before)
							}
							if !sameBits(bad, gradsBefore) {
								t.Fatal("refused Step changed gradients")
							}
							// A valid follow-up exposes any hidden velocity mutation.
							if err := opt.Step(params, good); err != nil {
								t.Fatal(err)
							}
							if err := ref.Step(want, good); err != nil {
								t.Fatal(err)
							}
							if !sameBits(params, want) {
								t.Fatalf("step after refusal changed optimizer state: got %v, want %v", params, want)
							}
						}
					})
				}
			}
		}
	}
}

func TestSGDOptimizerStepIgnoresUnrelatedNonFiniteGradients(t *testing.T) {
	opt, err := cudaautograd.NewSGDOptimizer(0.25, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	params := map[string][]float32{"w": {4}, "frozen": {3, -5}}
	grads := map[string][]float32{
		"w":         {2},
		"unrelated": {float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))},
	}
	for _, want := range []float32{3.5, 2.75} {
		if err := opt.Step(params, grads); err != nil {
			t.Fatal(err)
		}
		if params["w"][0] != want || !slices.Equal(params["frozen"], []float32{3, -5}) {
			t.Fatalf("Step = %v, want w=%g and unchanged frozen parameter", params, want)
		}
	}
}
