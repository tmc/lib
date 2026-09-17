package cudagroup

import "testing"

func TestLaunchValidation(t *testing.T) {
	warp, _ := New(Warp, Dim3{X: 32, Y: 1, Z: 1})
	if err := ValidateLaunch(LaunchRequirements{Grid: Dim3{X: 1, Y: 1, Z: 1}, Block: Dim3{X: 32, Y: 1, Z: 1}}, warp); err != nil {
		t.Fatal(err)
	}
	if err := ValidateLaunch(LaunchRequirements{Grid: Dim3{X: 1, Y: 1, Z: 1}, Block: Dim3{X: 16, Y: 1, Z: 1}}, warp); err == nil {
		t.Fatal("accepted mismatched block shape")
	}
	cluster, _ := New(Cluster, Dim3{X: 2, Y: 1, Z: 1})
	if err := ValidateLaunch(LaunchRequirements{Grid: Dim3{X: 1, Y: 1, Z: 1}, Block: Dim3{X: 32, Y: 1, Z: 1}}, cluster); err == nil {
		t.Fatal("accepted cluster without cluster dimensions")
	}
}

func TestFixtureLaunchShape(t *testing.T) {
	warp, _ := New(Warp, Dim3{X: 32, Y: 1, Z: 1})
	if err := ValidateLaunch(LaunchRequirements{Grid: Dim3{X: 1, Y: 1, Z: 1}, Block: Dim3{X: 32, Y: 1, Z: 1}}, warp); err != nil {
		t.Fatal(err)
	}
}

func TestNativeGroupReferenceMutants(t *testing.T) {
	warp, _ := New(Warp, Dim3{X: 32, Y: 1, Z: 1})
	fixture, err := GenerateReductionFixture(warp)
	if err != nil {
		t.Fatal(err)
	}
	if err := fixture.Collective.ValidateParticipation(31); err == nil {
		t.Fatal("dropped participant was accepted")
	}
	if _, err := GenerateReductionFixture(Group{kind: CTA, dims: Dim3{X: 32, Y: 1, Z: 1}}); err == nil {
		t.Fatal("cta substituted for warp fixture")
	}
}
