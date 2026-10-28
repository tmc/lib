package cudagreen

import (
	"fmt"
	"strings"
	"testing"
)

// TestGB10Groups reproduces the GB10 failure seen on dgx: 48 SMs, granule
// 8, and a driver that makes only 4 groups. A plan that ignores the limit
// needs 6 groups and fails as Open failed; a plan that models it fits.
func TestGB10Groups(t *testing.T) {
	driverGroups := []int{8, 8, 8, 8}

	blind, err := ComputePlan("sm_121a", 48, Granularity{Min: 8, Alignment: 8}, PrefillHeavy())
	if err != nil {
		t.Fatal(err)
	}
	_, err = assignGroups(blind, driverGroups)
	if err == nil || !strings.Contains(err.Error(), "driver made 4 groups of 8 SMs, plan needs 6") {
		t.Fatalf("assignGroups(blind plan) = %v, want the 4-groups error", err)
	}
	blind.Granularity.Groups = 4
	if err := VerifyPlan(blind); err == nil {
		t.Error("VerifyPlan accepted a 6-group plan on a 4-group device")
	}

	g := Granularity{Min: 8, Alignment: 8, Groups: len(driverGroups)}
	tests := []struct {
		name   string
		specs  []Spec
		sizes  string
		ranges string
	}{
		{"prefill", PrefillHeavy(), "[8 8 8 8]", "[[0 1] [1 2] [2 3] [3 4]]"},
		{"decode", DecodeHeavy(), "[8 8 8 8]", "[[0 1] [1 2] [2 3] [3 4]]"},
		{"jitter", []Spec{{Role: RoleCompute, Fraction: 0.75}, {Role: RoleDecode, Fraction: 0.25}}, "[24 8]", "[[0 3] [3 4]]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := ComputePlan("sm_121a", 48, g, tt.specs)
			if err != nil {
				t.Fatal(err)
			}
			if err := VerifyPlan(plan); err != nil {
				t.Fatal(err)
			}
			var sizes []int
			for _, q := range plan.Partitions {
				sizes = append(sizes, q.SMs)
			}
			if fmt.Sprint(sizes) != tt.sizes || plan.Unassigned != 16 {
				t.Errorf("sizes %v, unassigned %d; want %s, 16", sizes, plan.Unassigned, tt.sizes)
			}
			ranges, err := assignGroups(plan, driverGroups)
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprint(ranges) != tt.ranges {
				t.Errorf("ranges = %v, want %s", ranges, tt.ranges)
			}
		})
	}

	if _, err := ComputePlan("sm_121a", 48, g, []Spec{{MinSMs: 16}, {MinSMs: 16}, {MinSMs: 8}}); err == nil {
		t.Error("ComputePlan fit 5 groups on a 4-group device")
	}
}

func TestAssignGroupsErrors(t *testing.T) {
	plan, err := ComputePlan("sm_100a", 64, Granularity{Min: 8, Alignment: 8}, []Spec{{Fraction: 0.5}})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		groups []int
	}{
		{"too few", []int{8, 8, 8}},
		{"uneven", []int{8, 8, 16, 8}},
	}
	for _, tt := range tests {
		if _, err := assignGroups(plan, tt.groups); err == nil {
			t.Errorf("%s: assignGroups succeeded", tt.name)
		}
	}
	bad := plan
	bad.Partitions = append([]Partition(nil), plan.Partitions...)
	bad.Partitions[0].SMs = 12
	if _, err := assignGroups(bad, []int{8, 8, 8, 8}); err == nil {
		t.Error("assignGroups accepted a partition off the granule")
	}
}
