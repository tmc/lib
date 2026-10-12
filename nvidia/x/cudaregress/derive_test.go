package cudaregress_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

var hopper = cudaregress.Environment{
	Model:             "NVIDIA H100",
	ComputeCapability: "sm_90",
	Driver:            "550.54.14",
	CUDA:              "12.8",
}

func newDeriveParent(t *testing.T) *cudaregress.ExperimentPlan {
	t.Helper()
	dir := t.TempDir()
	src := makeTestArtifact(t, dir, "kernel.cu", "__global__ void k() {}")
	bin := makeTestArtifact(t, dir, "kernel.cubin", "CUBIN")
	in := makeTestArtifact(t, dir, "input.bin", "INPUT")
	plan, _, err := cudaregress.NewPlanFromIncident(validAdmittedIncident(), cudaregress.PlanOptions{
		Sources:  []cudaregress.ArtifactRef{src},
		Binaries: []cudaregress.ArtifactRef{bin},
		Inputs:   []cudaregress.InputRef{{Name: in.Name, Path: in.Path, Digest: in.Digest, ShapeDims: []int64{4, 8}}},
		Rounds:   3,
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

// The tests below through TestDerivePlanCopiesSlices compile against the
// pre-lineage API and fail at runtime on it.

func TestDerivePlanRefusesTamperedParent(t *testing.T) {
	parent := newDeriveParent(t)
	parent.StoppingRules.RelativeThreshold = 0.9 // edited after freeze
	if err := cudaregress.ValidatePlan(parent); err == nil {
		t.Fatal("ValidatePlan(tampered parent) = nil, want error")
	}
	d, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err == nil {
		t.Fatalf("DerivePlan(tampered parent) = plan with threshold %v, want error", d.StoppingRules.RelativeThreshold)
	}
}

func TestDerivePlanDeterministicID(t *testing.T) {
	parent := newDeriveParent(t)
	d1, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(time.Millisecond)
	d2, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	if d1.PlanID != d2.PlanID || d1.PlanDigest != d2.PlanDigest {
		t.Errorf("two derivations differ: (%s, %s) vs (%s, %s)", d1.PlanID, d1.PlanDigest, d2.PlanID, d2.PlanDigest)
	}
	if d1.PlanID == parent.PlanID {
		t.Errorf("derived PlanID = parent PlanID %q", parent.PlanID)
	}
	d3, err := cudaregress.DerivePlan(parent, hopper, "a different rationale")
	if err != nil {
		t.Fatal(err)
	}
	if d3.PlanID == d1.PlanID {
		t.Errorf("different rationale produced same PlanID %q", d1.PlanID)
	}
	other := hopper
	other.DeviceIndex = 1
	d4, err := cudaregress.DerivePlan(parent, other, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	if d4.PlanID == d1.PlanID {
		t.Errorf("different device produced same PlanID %q", d1.PlanID)
	}
}

func TestDerivePlanCopiesSlices(t *testing.T) {
	parent := newDeriveParent(t)
	d, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	want := d.PlanDigest
	parent.SourceArtifacts[0].Digest = "0000000000000000000000000000000000000000000000000000000000000000"
	parent.BinaryArtifacts[0].Path = "/nonexistent"
	parent.Inputs[0].ShapeDims[0] = 99
	parent.Inputs[0].Seed = 7
	got, err := cudaregress.ComputePlanDigest(d)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("mutating parent slices changed derived digest")
	}
	if err := cudaregress.ValidatePlan(d); err != nil {
		t.Errorf("ValidatePlan(derived) after parent mutation = %v, want nil", err)
	}
}

func TestDerivePlanRationaleRequired(t *testing.T) {
	parent := newDeriveParent(t)
	for _, r := range []string{"", "  \t"} {
		if _, err := cudaregress.DerivePlan(parent, hopper, r); err == nil {
			t.Errorf("DerivePlan(rationale %q) = nil error, want error", r)
		}
	}
}

// The tests below use the lineage API.

func TestDerivePlanLineage(t *testing.T) {
	parent := newDeriveParent(t)
	if parent.ParentPlanDigest != "" || parent.DerivationRationale != "" {
		t.Fatalf("root plan has lineage (%q, %q)", parent.ParentPlanDigest, parent.DerivationRationale)
	}
	const rationale = "evaluate on Hopper"
	d, err := cudaregress.DerivePlan(parent, hopper, rationale)
	if err != nil {
		t.Fatal(err)
	}
	if d.ParentPlanDigest != parent.PlanDigest {
		t.Errorf("ParentPlanDigest = %q, want %q", d.ParentPlanDigest, parent.PlanDigest)
	}
	if d.DerivationRationale != rationale {
		t.Errorf("DerivationRationale = %q, want %q", d.DerivationRationale, rationale)
	}
	if d.DeviceConfig.Model != hopper.Model || d.DeviceConfig.ComputeCapability != hopper.ComputeCapability {
		t.Errorf("DeviceConfig = %+v, want %+v", d.DeviceConfig, hopper)
	}
	if err := cudaregress.ValidatePlan(d); err != nil {
		t.Errorf("ValidatePlan(derived) = %v", err)
	}
	if err := cudaregress.VerifyDerivation(parent, d); err != nil {
		t.Errorf("VerifyDerivation = %v", err)
	}

	// A derivation of a derivation chains to its immediate parent.
	d2, err := cudaregress.DerivePlan(d, cudaregress.Environment{Model: "NVIDIA GB10", ComputeCapability: "sm_121a"}, "back to GB10")
	if err != nil {
		t.Fatal(err)
	}
	if d2.ParentPlanDigest != d.PlanDigest {
		t.Errorf("second-generation ParentPlanDigest = %q, want %q", d2.ParentPlanDigest, d.PlanDigest)
	}
	if err := cudaregress.VerifyDerivation(d, d2); err != nil {
		t.Errorf("VerifyDerivation(second generation) = %v", err)
	}
}

func TestDerivedPlanEditedAfterFreeze(t *testing.T) {
	tests := []struct {
		name string
		edit func(*cudaregress.ExperimentPlan)
	}{
		{"rationale", func(p *cudaregress.ExperimentPlan) { p.DerivationRationale = "something else" }},
		{"rationale cleared", func(p *cudaregress.ExperimentPlan) { p.DerivationRationale = "" }},
		{"parent digest", func(p *cudaregress.ExperimentPlan) {
			p.ParentPlanDigest = "1111111111111111111111111111111111111111111111111111111111111111"
		}},
		{"parent digest cleared", func(p *cudaregress.ExperimentPlan) { p.ParentPlanDigest = "" }},
		{"threshold", func(p *cudaregress.ExperimentPlan) { p.StoppingRules.RelativeThreshold = 0.9 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudaregress.DerivePlan(newDeriveParent(t), hopper, "evaluate on Hopper")
			if err != nil {
				t.Fatal(err)
			}
			tt.edit(d)
			if err := cudaregress.ValidatePlan(d); err == nil {
				t.Error("ValidatePlan(edited derived plan) = nil, want error")
			}
		})
	}
}

func TestValidatePlanLineageShape(t *testing.T) {
	tests := []struct {
		name      string
		parent    string
		rationale string
		ok        bool
	}{
		{"root", "", "", true},
		{"derived", "1111111111111111111111111111111111111111111111111111111111111111", "why", true},
		{"parent without rationale", "1111111111111111111111111111111111111111111111111111111111111111", "", false},
		{"rationale without parent", "", "why", false},
		{"short parent digest", "1111", "why", false},
		{"uppercase parent digest", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "why", false},
		{"blank rationale", "1111111111111111111111111111111111111111111111111111111111111111", " ", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newDeriveParent(t)
			p.ParentPlanDigest = tt.parent
			p.DerivationRationale = tt.rationale
			digest, err := cudaregress.ComputePlanDigest(p) // re-freeze so only the shape is tested
			if err != nil {
				t.Fatal(err)
			}
			p.PlanDigest = digest
			err = cudaregress.ValidatePlan(p)
			if (err == nil) != tt.ok {
				t.Errorf("ValidatePlan = %v, want ok=%v", err, tt.ok)
			}
		})
	}
}

func TestVerifyDerivationRejects(t *testing.T) {
	parent := newDeriveParent(t)
	d, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	otherParent := newDeriveParent(t)
	otherParent.Rounds = 2
	digest, err := cudaregress.ComputePlanDigest(otherParent)
	if err != nil {
		t.Fatal(err)
	}
	otherParent.PlanDigest = digest

	// A derived plan re-frozen to claim a different parent.
	reparented, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	reparented.ParentPlanDigest = otherParent.PlanDigest
	if reparented.PlanDigest, err = cudaregress.ComputePlanDigest(reparented); err != nil {
		t.Fatal(err)
	}

	// A derived plan whose non-device fields were changed and re-frozen.
	widened, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	widened.StoppingRules.RelativeThreshold = 0.9
	if widened.PlanDigest, err = cudaregress.ComputePlanDigest(widened); err != nil {
		t.Fatal(err)
	}

	// A derived plan edited without re-freezing.
	unfrozen, err := cudaregress.DerivePlan(parent, hopper, "evaluate on Hopper")
	if err != nil {
		t.Fatal(err)
	}
	unfrozen.StoppingRules.RelativeThreshold = 0.9

	tamperedParent := *parent
	tamperedParent.StoppingRules.RelativeThreshold = 0.9

	tests := []struct {
		name            string
		parent, derived *cudaregress.ExperimentPlan
		corrupted       bool   // errors.Is(err, ErrArtifactCorrupted)
		reason          string // substring of err
	}{
		{"wrong parent", otherParent, d, false, "names parent"},
		{"reparented", parent, reparented, false, "names parent"},
		{"widened after derive", parent, widened, false, "differs from derivation"},
		{"edited without refreeze", parent, unfrozen, true, "invalid derived plan"},
		{"root plan", parent, parent, false, "not derived"},
		{"tampered parent", &tamperedParent, d, true, "invalid parent plan"},
		{"nil parent", nil, d, false, "nil experiment plan"},
		{"nil derived", parent, nil, false, "nil experiment plan"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := cudaregress.VerifyDerivation(tt.parent, tt.derived)
			if err == nil {
				t.Fatal("VerifyDerivation = nil, want error")
			}
			if !strings.Contains(err.Error(), tt.reason) {
				t.Errorf("VerifyDerivation = %v, want reason %q", err, tt.reason)
			}
			if got := errors.Is(err, cudaregress.ErrArtifactCorrupted); got != tt.corrupted {
				t.Errorf("errors.Is(%v, ErrArtifactCorrupted) = %v, want %v", err, got, tt.corrupted)
			}
		})
	}
}

func TestReplayForkRecordsParent(t *testing.T) {
	parent := newDeriveParent(t)
	runner := cudaregress.ReplayFunc(func(ctx context.Context, arm string, round int) (cudatrial.Receipt, []float64, error) {
		return cudatrial.Receipt{Complete: true, Name: arm, TimingKind: cudatrial.TimingWallElapsed, WallNs: 1000, WallElapsed: time.Microsecond}, nil, nil
	})
	const rationale = "evaluate on Hopper"
	exec, _, err := cudaregress.Replay(context.Background(), parent, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: hopper,
		AllowDriftFork:     true,
		DriftRationale:     rationale,
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := cudaregress.DerivePlan(parent, hopper, rationale)
	if err != nil {
		t.Fatal(err)
	}
	if exec.ParentPlanDigest != parent.PlanDigest {
		t.Errorf("receipt ParentPlanDigest = %q, want %q", exec.ParentPlanDigest, parent.PlanDigest)
	}
	if exec.PlanID != want.PlanID || exec.PlanDigest != want.PlanDigest {
		t.Errorf("receipt plan = (%s, %s), want (%s, %s)", exec.PlanID, exec.PlanDigest, want.PlanID, want.PlanDigest)
	}

	// Without drift the receipt carries the plan's own (empty) lineage.
	exec, _, err = cudaregress.Replay(context.Background(), parent, runner, cudaregress.ReplayOptions{
		CurrentEnvironment: validAdmittedIncident().ObservedEnvironment,
	})
	if err != nil {
		t.Fatal(err)
	}
	if exec.ParentPlanDigest != "" {
		t.Errorf("root replay ParentPlanDigest = %q, want empty", exec.ParentPlanDigest)
	}
}

func ExampleVerifyDerivation() {
	parent, _, err := cudaregress.NewPlanFromIncident(cudaregress.Incident{
		ID:       "inc-001",
		Admitted: true,
		ObservedEnvironment: cudaregress.Environment{
			Model:             "NVIDIA GB10",
			ComputeCapability: "sm_121a",
		},
	}, cudaregress.PlanOptions{Rounds: 3})
	if err != nil {
		fmt.Println(err)
		return
	}
	derived, err := cudaregress.DerivePlan(parent, cudaregress.Environment{
		Model:             "NVIDIA H100",
		ComputeCapability: "sm_90",
	}, "evaluate on Hopper")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(derived.ParentPlanDigest == parent.PlanDigest)
	fmt.Println(derived.DerivationRationale)
	fmt.Println(cudaregress.VerifyDerivation(parent, derived))
	// Output:
	// true
	// evaluate on Hopper
	// <nil>
}
