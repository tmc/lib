package cuptirange

import (
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

// TestLayoutShapes pins each measured parameter-struct shape to the CUPTI
// versions it was read from. The two structs are listed separately because they
// did not grow together: 130101 extends the properties struct and leaves the
// initialize struct at its 130001 shape. An unmeasured version takes the shape
// of the newest measured one below it in the same major; any other fails closed.
func TestLayoutShapes(t *testing.T) {
	for _, tt := range []struct {
		version    uint32
		qualified  bool
		singlePass bool // Initialize_Params ends at pSinglePassMetricSetName.
		scoped     bool // GetMetricProperties_Params ends at metricCollectionScope.
	}{
		{version: 24, qualified: true},
		{version: 26, qualified: true},
		{version: 28, qualified: true},
		{version: 130001, qualified: true},
		{version: 130101, qualified: true, scoped: true},
		{version: 130200, qualified: true, singlePass: true, scoped: true},
		{version: 130202, qualified: true, singlePass: true, scoped: true},
		{version: 130301, qualified: true, singlePass: true, scoped: true},
		{version: 130400, qualified: true, singlePass: true, scoped: true},
		{version: 0},
		{version: 20},
		{version: 25, qualified: true},
		{version: 27, qualified: true},
		{version: 29, qualified: true},
		{version: 120901},
		{version: 130000},
		{version: 130002, qualified: true},
		{version: 130100, qualified: true},
		{version: 130201, qualified: true, singlePass: true, scoped: true},
		{version: 130300, qualified: true, singlePass: true, scoped: true},
		{version: 130401, qualified: true, singlePass: true, scoped: true},
		{version: 130500, qualified: true, singlePass: true, scoped: true},
		{version: 140000},
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

// TestGeneratedVersionIsQualified pins the qualification table to the version
// the Go types were generated from. Here that is a claim about shapes and not
// only about admission: the two final fields the shape predicates report on are
// fields of the generated Go structs, so both must hold at the generated
// version or the structs and the predicates describe different headers.
//
// The table holds version literals, so nothing otherwise connects it to
// [cupti.ApiVersion] -- a regeneration moves the constant and leaves every
// literal, and neither the compiler nor any other test notices. Whatever else
// the table admits, it must admit the version whose struct shapes these types
// actually are.
func TestGeneratedVersionIsQualified(t *testing.T) {
	if !qualifiedLayout(cupti.ApiVersion) {
		t.Errorf("qualifiedLayout(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
	if !singlePassInitialize(cupti.ApiVersion) {
		t.Errorf("singlePassInitialize(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
	if !scopedProperties(cupti.ApiVersion) {
		t.Errorf("scopedProperties(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
}
