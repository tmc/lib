package cudaquant

import "testing"

func TestNormNativeTargets(t *testing.T) {
	for _, tc := range []struct {
		target       string
		major, minor int
	}{
		{"sm_90", 9, 0},
		{"sm_90a", 9, 0},
		{"sm_100a", 10, 0},
		{"sm_121a", 12, 1},
	} {
		t.Run(tc.target, func(t *testing.T) {
			major, minor, ok := normNativeCapability(tc.target)
			if !ok || major != tc.major || minor != tc.minor {
				t.Fatalf("capability(%q) = %d.%d, %v; want %d.%d, true", tc.target, major, minor, ok, tc.major, tc.minor)
			}
		})
	}
	for _, target := range []string{"", "compute_90", "sm_80", "sm_90f", "sm_900", "sm_100", "sm_121", "sm_90a\x00"} {
		t.Run("unsupported/"+target, func(t *testing.T) {
			if _, _, ok := normNativeCapability(target); ok {
				t.Fatalf("accepted unsupported target %q", target)
			}
		})
	}
}

func TestNormNativeDeviceMatches(t *testing.T) {
	for _, tc := range []struct {
		name, target string
		major, minor int
		want         bool
	}{
		{"hopper", "sm_90", 9, 0, true},
		{"hopper_a", "sm_90a", 9, 0, true},
		{"blackwell", "sm_100a", 10, 0, true},
		{"blackwell_121", "sm_121a", 12, 1, true},
		{"older_device", "sm_90", 8, 0, false},
		{"newer_device", "sm_90", 10, 0, false},
		{"different_minor", "sm_90a", 9, 1, false},
		{"blackwell_on_hopper", "sm_100a", 9, 0, false},
		{"different_blackwell_minor", "sm_121a", 12, 0, false},
		{"virtual_target", "compute_90", 9, 0, false},
		{"unsupported_target", "sm_80", 8, 0, false},
		{"invalid_target", "sm_bad", 9, 0, false},
		{"invalid_capability", "sm_90", 0, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := normNativeDeviceMatches(tc.target, tc.major, tc.minor); got != tc.want {
				t.Fatalf("matches(%q, %d, %d) = %v; want %v", tc.target, tc.major, tc.minor, got, tc.want)
			}
		})
	}
}
