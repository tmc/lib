package cudafuse

import "testing"

// TestShapeElements checks that every dimension is validated regardless of
// where a zero appears, and that a zero dimension makes the count exactly
// zero rather than an overflow.
func TestShapeElements(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		want    uint64
		wantErr bool
	}{
		{"nil_is_scalar", nil, 1, false},
		{"rank0_scalar", []int{}, 1, false},
		{"empty", []int{0}, 0, false},
		{"empty_middle", []int{2, 0, 3}, 0, false},
		{"rank2", []int{3, 4}, 12, false},
		{"large_fits", []int{1 << 31, 1 << 31}, 1 << 62, false},
		{"zero_then_negative", []int{0, -1}, 0, true},
		{"negative_then_zero", []int{-1, 0}, 0, true},
		{"zero_then_negative_later", []int{2, 0, 3, -4}, 0, true},
		{"negative", []int{2, -3}, 0, true},
		{"overflow", []int{1 << 32, 1 << 32}, 0, true},
		{"overflow_then_negative", []int{1 << 32, 1 << 32, -1}, 0, true},
		{"zero_then_overflow", []int{0, 1 << 62, 1 << 62}, 0, false},
		{"overflow_then_zero", []int{1 << 62, 1 << 62, 0}, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := shapeElements(tt.shape)
			if (err != nil) != tt.wantErr {
				t.Fatalf("shapeElements(%v) = %d, %v; want error %v", tt.shape, got, err, tt.wantErr)
			}
			if err == nil && got != tt.want {
				t.Fatalf("shapeElements(%v) = %d, want %d", tt.shape, got, tt.want)
			}
		})
	}
}
