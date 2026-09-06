//go:build !windows

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
		if got, err := PadTo(tt.n, tt.gran); err != nil || got != tt.want {
			t.Errorf("PadTo(%d, %d) = %d, want %d", tt.n, tt.gran, got, tt.want)
		}
	}
}

func TestPadToErrors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		n, gran uintptr
	}{{"zero granularity", 1, 0}, {"overflow", ^uintptr(0), 2}} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := PadTo(tt.n, tt.gran); err == nil {
				t.Fatal("PadTo succeeded")
			}
		})
	}
}
