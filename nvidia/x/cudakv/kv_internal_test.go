package cudakv

import "testing"

func TestValidatePagedDimensions(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name                                  string
		layers, heads, dim, blockSize, blocks int
	}{
		{"valid", 2, 4, 64, 16, 64},
		{"layers", -1, 4, 64, 16, 64},
		{"heads", 2, 0, 64, 16, 64},
		{"head dimension", 2, 4, 0, 16, 64},
		{"block size", 2, 4, 64, 0, 64},
		{"blocks", 2, 4, 64, 16, 0},
		{"per-layer elements overflow", 1, 2, 2, 2, maxInt},
		{"aggregate layer overflow", maxInt/2 + 1, 1, 1, 2, 1},
		{"byte width overflow", 1, 1, 1, 1, maxInt/4 + 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validatePagedDimensions(test.layers, test.heads, test.dim, test.blockSize, test.blocks)
			if (err == nil) != (test.name == "valid") {
				t.Fatalf("validatePagedDimensions error = %v", err)
			}
		})
	}
}
