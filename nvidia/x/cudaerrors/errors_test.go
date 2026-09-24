package cudaerrors_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// TestSentinelsAreDistinct asserts the two answers cannot be matched for each
// other. A package that wrapped one in the other would make errors.Is report
// an unprobed target as a refused one.
func TestSentinelsAreDistinct(t *testing.T) {
	if errors.Is(cudaerrors.ErrHardwarePending, cudaerrors.ErrUnsupported) {
		t.Error("ErrHardwarePending matches ErrUnsupported")
	}
	if errors.Is(cudaerrors.ErrUnsupported, cudaerrors.ErrHardwarePending) {
		t.Error("ErrUnsupported matches ErrHardwarePending")
	}
}

// TestWrappingCarriesCallerContext shows the intended use: the sentinel says
// which answer it is, the wrap says who was asking.
func TestWrappingCarriesCallerContext(t *testing.T) {
	err := fmt.Errorf("cudacluster: no feature record for sm_98a: %w", cudaerrors.ErrHardwarePending)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatal("wrapped sentinel does not match")
	}
	if errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatal("wrapped ErrHardwarePending matches ErrUnsupported")
	}
}
