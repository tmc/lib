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

// TestMaxCallersIsTheDocumentedBound ties the constant to the number the
// external test builds a call chain against. The truncation cannot be
// tested from inside this package, because a frame of this package is
// skipped as the capture's own.
func TestMaxCallersIsTheDocumentedBound(t *testing.T) {
	if maxCallers != 8 {
		t.Errorf("maxCallers = %d; TestCallersAreTruncated builds its chain against 8", maxCallers)
	}
}
