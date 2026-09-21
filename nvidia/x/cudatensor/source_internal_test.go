package cudatensor

import (
	"strings"
	"testing"
)

// TestPkgPrefixIsResolved guards the mechanism the capture depends on. An
// empty prefix makes captureSource return the zero location for every
// operation, which every assertion above would report as a missing location --
// but only if this package is still where the prefix computation thinks it is.
func TestPkgPrefixIsResolved(t *testing.T) {
	if pkgPrefix == "" {
		t.Fatal("pkgPrefix is empty: every captured location would be absent")
	}
	if !strings.HasSuffix(pkgPrefix, "/cudatensor.") {
		t.Errorf("pkgPrefix = %q, want it to end in /cudatensor.", pkgPrefix)
	}
}
