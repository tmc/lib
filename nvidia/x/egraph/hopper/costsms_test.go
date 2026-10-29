package hopper

import "testing"

// TestCostConfigDefaultSMs pins the SM count a CostConfig defaults to for
// each device or architecture it can resolve a ceiling for. An
// architecture without a product takes its first listed product's count;
// sm_70 has no profile and keeps the H200's.
func TestCostConfigDefaultSMs(t *testing.T) {
	tests := []struct {
		device string
		want   int
	}{
		{"", 132},
		{"NVIDIA H200", 132},
		{"NVIDIA GB10", 48},
		{"sm_70", 132},
		{"sm_75", 40},
		{"sm_80", 108},
		{"sm_86", 72},
		{"sm_89", 142},
		{"sm_90", 132},
		{"sm_121", 48},
		{"sm_121a", 48},
	}
	for _, tt := range tests {
		c, err := CostConfig{Device: tt.device}.norm()
		if err != nil {
			t.Errorf("%q: %v", tt.device, err)
			continue
		}
		if c.SMs != tt.want {
			t.Errorf("%q: SMs = %d, want %d", tt.device, c.SMs, tt.want)
		}
	}
}
