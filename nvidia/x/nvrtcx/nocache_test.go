package nvrtcx

import "testing"

func TestHasNVRTCCache(t *testing.T) {
	tests := []struct {
		major, minor int32
		want         bool
	}{
		{11, 8, false},
		{12, 2, false},
		{12, 8, false},
		{12, 9, true},
		{13, 0, true},
	}
	for _, tt := range tests {
		if got := hasNVRTCCache(tt.major, tt.minor); got != tt.want {
			t.Errorf("hasNVRTCCache(%d, %d) = %v, want %v", tt.major, tt.minor, got, tt.want)
		}
	}
}
