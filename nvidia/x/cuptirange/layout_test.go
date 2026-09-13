package cuptirange

import "testing"

// TestLayoutFamilies pins the two measured parameter-struct shapes to the CUPTI
// versions they were read from. A version not listed here must fail closed
// rather than inherit a neighbour's struct size.
func TestLayoutFamilies(t *testing.T) {
	for _, tt := range []struct {
		version   uint32
		qualified bool
		scoped    bool
	}{
		{27, true, false},
		{130001, true, false},
		{130200, true, true},
		{130400, true, true},
		{0, false, false},
		{28, false, false},
		{120901, false, false},
		{130002, false, false},
		{130300, false, false},
		{130500, false, false},
	} {
		if got := qualifiedLayout(tt.version); got != tt.qualified {
			t.Errorf("qualifiedLayout(%d) = %v, want %v", tt.version, got, tt.qualified)
		}
		if got := scopedLayout(tt.version); got != tt.scoped {
			t.Errorf("scopedLayout(%d) = %v, want %v", tt.version, got, tt.scoped)
		}
	}
}
