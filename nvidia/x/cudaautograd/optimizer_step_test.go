package cudaautograd_test

import (
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
)

// stepFixture returns params p0..p{n-1}, each with a matching gradient,
// plus a parameter "bad" whose gradient has the wrong length.
func stepFixture(n int) (params, grads map[string][]float32) {
	params = map[string][]float32{"bad": {1, 2, 3}}
	grads = map[string][]float32{"bad": {1, 2}}
	for i := range n {
		name := fmt.Sprintf("p%d", i)
		params[name] = []float32{float32(i) + 0.5, -float32(i) - 0.25}
		grads[name] = []float32{0.75, -1.5}
	}
	return params, grads
}

func cloneParams(m map[string][]float32) map[string][]float32 {
	c := make(map[string][]float32, len(m))
	for k, v := range m {
		c[k] = slices.Clone(v)
	}
	return c
}

func sameBits(a, b map[string][]float32) bool {
	return maps.EqualFunc(a, b, func(x, y []float32) bool {
		return slices.EqualFunc(x, y, func(p, q float32) bool {
			return math.Float32bits(p) == math.Float32bits(q)
		})
	})
}

// TestSGDOptimizerStepIsAllOrNothing checks that a length mismatch on any
// parameter leaves every parameter and the velocity state untouched,
// whatever order the params map is iterated in.
func TestSGDOptimizerStepIsAllOrNothing(t *testing.T) {
	tests := []struct {
		name     string
		momentum float32
		params   int
	}{
		{"two params without momentum", 0, 1},
		{"two params with momentum", 0.5, 1},
		{"many params without momentum", 0, 8},
		{"many params with momentum", 0.5, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for iter := range 64 {
				opt, err := cudaautograd.NewSGDOptimizer(0.25, tt.momentum)
				if err != nil {
					t.Fatal(err)
				}
				ref, err := cudaautograd.NewSGDOptimizer(0.25, tt.momentum)
				if err != nil {
					t.Fatal(err)
				}
				params, grads := stepFixture(tt.params)
				refParams := cloneParams(params)
				goodGrads := cloneParams(grads)
				goodGrads["bad"] = []float32{0, 0, 0}

				// Build velocity on both optimizers with one good step.
				if err := opt.Step(params, goodGrads); err != nil {
					t.Fatal(err)
				}
				if err := ref.Step(refParams, goodGrads); err != nil {
					t.Fatal(err)
				}
				before := cloneParams(params)

				err = opt.Step(params, grads)
				if err == nil || !strings.HasPrefix(err.Error(), "cudaautograd: ") || !strings.Contains(err.Error(), `"bad"`) {
					t.Fatalf("iteration %d: Step with mismatched gradient = %v, want cudaautograd error naming \"bad\"", iter, err)
				}
				if !sameBits(params, before) {
					t.Fatalf("iteration %d: failed Step changed params:\n got %v\nwant %v", iter, params, before)
				}

				// The velocity must also be unchanged: the next good step
				// must match an optimizer that never saw the failed step.
				if err := opt.Step(params, goodGrads); err != nil {
					t.Fatal(err)
				}
				if err := ref.Step(refParams, goodGrads); err != nil {
					t.Fatal(err)
				}
				if !sameBits(params, refParams) {
					t.Fatalf("iteration %d: step after failed Step diverged from reference:\n got %v\nwant %v", iter, params, refParams)
				}
			}
		})
	}
}

func TestSGDOptimizerStepRefusals(t *testing.T) {
	valid, err := cudaautograd.NewSGDOptimizer(0.25, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		opt     *cudaautograd.SGDOptimizer
		params  map[string][]float32
		grads   map[string][]float32
		wantErr string
	}{
		{"nil optimizer", nil, map[string][]float32{"w": {1}}, map[string][]float32{"w": {1}}, "cudaautograd: nil optimizer"},
		{"nil params", valid, nil, map[string][]float32{"w": {1}}, "cudaautograd: nil params map"},
		{"nil grads", valid, map[string][]float32{"w": {1}}, nil, "cudaautograd: nil grads map"},
		{"short gradient", valid, map[string][]float32{"w": {1, 2}}, map[string][]float32{"w": {1}}, `cudaautograd: param "w" len 2 != grad len 1`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := cloneParams(tt.params)
			err := tt.opt.Step(tt.params, tt.grads)
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("Step = %v, want %q", err, tt.wantErr)
			}
			if !sameBits(tt.params, before) {
				t.Fatalf("failed Step changed params: got %v, want %v", tt.params, before)
			}
		})
	}
}

// TestSGDOptimizerStepSkipsParamsWithoutGradient checks that a parameter
// absent from grads is neither refused nor updated, and that Step does not
// start a velocity for it.
func TestSGDOptimizerStepSkipsParamsWithoutGradient(t *testing.T) {
	opt, err := cudaautograd.NewSGDOptimizer(0.25, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	params := map[string][]float32{"w": {1, 2}, "frozen": {3, -4, 5}}
	for i := range 2 {
		if err := opt.Step(params, map[string][]float32{"w": {1, 1}}); err != nil {
			t.Fatalf("step %d: Step without gradient for frozen = %v", i, err)
		}
	}
	if want := []float32{3, -4, 5}; !slices.Equal(params["frozen"], want) {
		t.Fatalf("Step changed frozen to %v, want %v", params["frozen"], want)
	}
	// w: velocity 0.25 then 0.375, so w = {1, 2} - 0.625.
	if want := []float32{0.375, 1.375}; !slices.Equal(params["w"], want) {
		t.Fatalf("w = %v, want %v", params["w"], want)
	}

	// The first gradient for frozen must see zero velocity: the result
	// matches a fresh optimizer's first step.
	grad := map[string][]float32{"frozen": {2, 2, 2}}
	if err := opt.Step(params, grad); err != nil {
		t.Fatal(err)
	}
	fresh, err := cudaautograd.NewSGDOptimizer(0.25, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]float32{"frozen": {3, -4, 5}}
	if err := fresh.Step(want, grad); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(params["frozen"], want["frozen"]) {
		t.Fatalf("first update of frozen = %v, want %v", params["frozen"], want["frozen"])
	}
}

func TestSGDOptimizerStepZeroValue(t *testing.T) {
	var opt cudaautograd.SGDOptimizer
	params := map[string][]float32{"w": {1}}
	if err := opt.Step(params, map[string][]float32{"w": {2}}); err != nil {
		t.Fatal(err)
	}
	if params["w"][0] != 1 {
		t.Fatalf("zero-value Step changed w to %g, want 1", params["w"][0])
	}
}

func TestTrainingStepNilLossNode(t *testing.T) {
	f := newCheckFixture(t)
	opt, err := cudaautograd.NewSGDOptimizer(0.25, 0)
	if err != nil {
		t.Fatal(err)
	}
	params := map[string][]float32{"x": {2, -3}}
	before := cloneParams(params)
	loss, out, err := cudaautograd.TrainingStep(f.graph, nil, f.result, opt, params, nil)
	if err == nil || err.Error() != "cudaautograd: nil loss node" {
		t.Fatalf("TrainingStep(nil loss) = %g, %v, %v; want cudaautograd: nil loss node", loss, out, err)
	}
	if loss != 0 || out != nil {
		t.Fatalf("TrainingStep(nil loss) = %g, %v; want 0, nil", loss, out)
	}
	if !sameBits(params, before) {
		t.Fatalf("TrainingStep(nil loss) changed params: got %v, want %v", params, before)
	}
}
