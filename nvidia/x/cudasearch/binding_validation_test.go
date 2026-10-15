package cudasearch

import (
	"context"
	"errors"
	"testing"
)

func TestWorkloadRequiredIdentities(t *testing.T) {
	for _, test := range []struct {
		name            string
		names           []string
		inputs, weights map[string]string
		valid           bool
	}{
		{"input", []string{"x"}, map[string]string{"x": "input-v1"}, nil, true},
		{"input_and_weight", []string{"x", "w"}, map[string]string{"x": "input-v1"}, map[string]string{"w": "weight-v1"}, true},
		{"matching_roles", []string{"x"}, map[string]string{"x": "v1"}, map[string]string{"x": "v1"}, true},
		{"missing_all", []string{"x"}, nil, nil, false},
		{"missing_second", []string{"x", "w"}, map[string]string{"x": "input-v1"}, nil, false},
		{"unrelated", []string{"x"}, map[string]string{"y": "v1"}, nil, false},
		{"empty_input_identity", []string{"x"}, map[string]string{"x": ""}, nil, false},
		{"empty_weight_identity", []string{"w"}, nil, map[string]string{"w": ""}, false},
		{"empty_identity_name", []string{"x"}, map[string]string{"x": "v1", "": "v2"}, nil, false},
		{"empty_required_name", []string{""}, map[string]string{"x": "v1"}, nil, false},
		{"duplicate_required_name", []string{"x", "x"}, map[string]string{"x": "v1"}, nil, false},
		{"conflicting_roles", []string{"x"}, map[string]string{"x": "v1"}, map[string]string{"x": "v2"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			b := WorkloadBindings{Name: "bound-fixture", InputNames: test.names, InputIdentities: test.inputs, WeightIdentities: test.weights, NumericalPolicy: NumericalPolicyExact}
			err := b.Validate()
			if (err == nil) != test.valid || err != nil && !errors.Is(err, ErrMissingBinding) {
				t.Fatalf("Validate() = %v, valid %v", err, test.valid)
			}
		})
	}
}

func TestExecutableMissingBindingBeforeWork(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*WorkloadBindings)
	}{
		{"missing_input", func(b *WorkloadBindings) { b.InputIdentities = nil }},
		{"missing_weight", func(b *WorkloadBindings) { b.InputNames = append(b.InputNames, "weight") }},
		{"empty_identity", func(b *WorkloadBindings) { b.InputIdentities["x"] = "" }},
		{"conflicting_weight", func(b *WorkloadBindings) { b.WeightIdentities = map[string]string{"x": "other"} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			e, base, compiled, order := executableFixture(t)
			test.mutate(&e.Workload)
			leaseChecks := 0
			e.HardwareLockChecker = func() bool { leaseChecks++; return true }
			candidate, err := e.Evaluate(context.Background(), executableState(), 0)
			if candidate != nil || !errors.Is(err, ErrMissingBinding) {
				t.Fatalf("missing binding admitted: candidate %v, error %v", candidate, err)
			}
			if len(*order) != 0 || base.closed != 0 || compiled.closed != 0 || leaseChecks != 0 {
				t.Fatalf("missing binding reached executable work: order=%v lease checks=%d", *order, leaseChecks)
			}
		})
	}
}
