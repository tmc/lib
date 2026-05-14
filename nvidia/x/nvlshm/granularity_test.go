package nvlshm

import "testing"

func TestPadTo(t *testing.T) {
	tests := []struct {
		n, gran, want uintptr
	}{
		{0, 4096, 0},
		{1, 4096, 4096},
		{4096, 4096, 4096},
		{4097, 4096, 8192},
		{1 << 21, 1 << 21, 1 << 21},
		{(1 << 21) + 1, 1 << 21, 2 << 21},
		{100, 1, 100},
	}
	for _, tt := range tests {
		if got := PadTo(tt.n, tt.gran); got != tt.want {
			t.Errorf("PadTo(%d, %d) = %d, want %d", tt.n, tt.gran, got, tt.want)
		}
	}
}

func TestPadToZeroGranularityPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("PadTo(1, 0): want panic")
		}
	}()
	PadTo(1, 0)
}
