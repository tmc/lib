package cuptirange

import "testing"

// TestLayoutShapes pins each measured parameter-struct shape to the CUPTI
// versions it was read from. The two structs are listed separately because they
// did not grow together: 130101 extends the properties struct and leaves the
// initialize struct at its 130001 shape. A version not listed here must fail
// closed rather than inherit a neighbour's struct size.
func TestLayoutShapes(t *testing.T) {
	for _, tt := range []struct {
		version    uint32
		qualified  bool
		singlePass bool // Initialize_Params ends at pSinglePassMetricSetName.
		scoped     bool // GetMetricProperties_Params ends at metricCollectionScope.
	}{
		{version: 27, qualified: true},
		{version: 130001, qualified: true},
		{version: 130101, qualified: true, scoped: true},
		{version: 130200, qualified: true, singlePass: true, scoped: true},
		{version: 130202, qualified: true, singlePass: true, scoped: true},
		{version: 130301, qualified: true, singlePass: true, scoped: true},
		{version: 130400, qualified: true, singlePass: true, scoped: true},
		{version: 0},
		{version: 28},
		{version: 120901},
		{version: 130002},
		{version: 130100},
		{version: 130201},
		{version: 130300},
		{version: 130500},
	} {
		if got := qualifiedLayout(tt.version); got != tt.qualified {
			t.Errorf("qualifiedLayout(%d) = %v, want %v", tt.version, got, tt.qualified)
		}
		if got := singlePassInitialize(tt.version); got != tt.singlePass {
			t.Errorf("singlePassInitialize(%d) = %v, want %v", tt.version, got, tt.singlePass)
		}
		if got := scopedProperties(tt.version); got != tt.scoped {
			t.Errorf("scopedProperties(%d) = %v, want %v", tt.version, got, tt.scoped)
		}
		if !tt.qualified && (tt.singlePass || tt.scoped) {
			t.Errorf("%d: an unqualified version cannot have a shape", tt.version)
		}
	}
}
