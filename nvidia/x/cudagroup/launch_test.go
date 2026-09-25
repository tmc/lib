package cudagroup

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestLaunchValidation(t *testing.T) {
	warp, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	if err := ValidateLaunch(LaunchRequirements{Grid: driver.Dim3{X: 1, Y: 1, Z: 1}, Block: driver.Dim3{X: 32, Y: 1, Z: 1}}, warp); err != nil {
		t.Fatal(err)
	}
	if err := ValidateLaunch(LaunchRequirements{Grid: driver.Dim3{X: 1, Y: 1, Z: 1}, Block: driver.Dim3{X: 16, Y: 1, Z: 1}}, warp); err == nil {
		t.Fatal("accepted mismatched block shape")
	}
	cluster, _ := New(Cluster, driver.Dim3{X: 2, Y: 1, Z: 1})
	if err := ValidateLaunch(LaunchRequirements{Grid: driver.Dim3{X: 1, Y: 1, Z: 1}, Block: driver.Dim3{X: 32, Y: 1, Z: 1}}, cluster); err == nil {
		t.Fatal("accepted cluster without cluster dimensions")
	}
}

func TestFixtureLaunchShape(t *testing.T) {
	warp, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	if err := ValidateLaunch(LaunchRequirements{Grid: driver.Dim3{X: 1, Y: 1, Z: 1}, Block: driver.Dim3{X: 32, Y: 1, Z: 1}}, warp); err != nil {
		t.Fatal(err)
	}
}

func TestNativeGroupReferenceMutants(t *testing.T) {
	warp, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	fixture, err := GenerateReductionFixture(warp)
	if err != nil {
		t.Fatal(err)
	}
	if err := fixture.Collective.ValidateParticipation(31); err == nil {
		t.Fatal("dropped participant was accepted")
	}
	if _, err := GenerateReductionFixture(Group{kind: CTA, dims: driver.Dim3{X: 32, Y: 1, Z: 1}}); err == nil {
		t.Fatal("cta substituted for warp fixture")
	}
}
